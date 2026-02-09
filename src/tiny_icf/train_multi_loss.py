"""Enhanced training script with multi-loss support."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_icf.data import (
    WordICFDataset,
    compute_normalized_icf,
    load_frequency_list,
    stratified_sample,
)
from tiny_icf.loss import CombinedLoss
from tiny_icf.loss_multi import EnhancedMultiLoss, CurriculumMultiLoss
from tiny_icf.model import UniversalICF
from tiny_icf.train import generate_ranking_pairs


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_common_rare_indices(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Identify common (ICF < 0.3) and rare (ICF > 0.7) words in batch.

    Returns:
        (common_indices, rare_indices) as tensors
    """
    if targets.dim() > 1 and targets.size(1) == 1:
        targets_flat = targets.squeeze(1)
    else:
        targets_flat = targets

    common_mask = targets_flat < 0.3
    rare_mask = targets_flat > 0.7

    common_indices = torch.where(common_mask)[0]
    rare_indices = torch.where(rare_mask)[0]

    return common_indices, rare_indices


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_multi_loss: bool = False,
) -> float:
    """Train for one epoch with optional multi-loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for byte_tensors, icf_targets in tqdm(dataloader, desc="Training"):
        byte_tensors = byte_tensors.to(device)
        icf_targets = icf_targets.to(device)

        optimizer.zero_grad()
        predictions = model(byte_tensors)

        if use_multi_loss and isinstance(criterion, EnhancedMultiLoss):
            # Generate pairs for ranking loss with weighted sampling
            # Increase pairs for better ranking signal
            # Use min_diff to only learn from pairs with meaningful ICF difference
            n_pairs = min(len(icf_targets), 32)  # More pairs for better ranking learning
            pairs, pair_diffs = generate_ranking_pairs(
                icf_targets, n_pairs, min_diff=0.05, use_weighted_sampling=True
            )

            # Get common/rare indices for contrastive loss
            common_indices, rare_indices = get_common_rare_indices(icf_targets)

            # Compute multi-loss with smooth rewards
            loss = criterion(
                predictions,
                icf_targets,
                pairs=pairs if len(pairs) > 0 else None,
                common_indices=common_indices if len(common_indices) > 0 else None,
                rare_indices=rare_indices if len(rare_indices) > 0 else None,
                word_similarity=None,  # Optional: could compute from byte sequences
                pair_target_diffs=pair_diffs if len(pair_diffs) > 0 else None,
            )
        else:
            # Standard loss (CombinedLoss)
            # Increase pairs for better ranking signal
            # Use min_diff to only learn from pairs with meaningful ICF difference
            n_pairs = min(len(icf_targets), 32)  # More pairs for better ranking learning
            pairs, pair_diffs = generate_ranking_pairs(
                icf_targets, n_pairs, min_diff=0.05, use_weighted_sampling=True
            )
            loss = criterion(
                predictions,
                icf_targets,
                pairs=pairs,
                pair_target_diffs=pair_diffs,
                smooth_ranking=True,
            )

        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for byte_tensors, icf_targets in tqdm(dataloader, desc="Validating"):
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)

            predictions = model(byte_tensors)

            # For validation, just use Huber loss (simpler)
            if isinstance(criterion, (EnhancedMultiLoss, CurriculumMultiLoss)):
                # Use only Huber component for validation
                from tiny_icf.loss import huber_loss

                loss = huber_loss(predictions, icf_targets)
            else:
                loss = criterion(predictions, icf_targets)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train Universal ICF model with multi-loss")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=20, help="Max word length")
    parser.add_argument("--augment-prob", type=float, default=0.1, help="Augmentation probability")
    parser.add_argument(
        "--output", type=Path, default=Path("models/model_multi_loss.pt"), help="Output model path"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--multi-loss", action="store_true", help="Use enhanced multi-loss")
    parser.add_argument(
        "--curriculum", action="store_true", help="Use curriculum multi-loss (progressive)"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(42)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print("Random seed: 42 (reproducible)")
    print(f"Multi-loss: {args.multi_loss}")
    print(f"Curriculum: {args.curriculum}")

    # Load data
    print("Loading frequency list...")
    try:
        word_counts, total_tokens = load_frequency_list(args.data)
        print(f"Loaded {len(word_counts)} words, {total_tokens:,} total tokens")
    except Exception as e:
        print(f"Error loading frequency list: {e}")
        raise

    # Compute ICF
    print("Computing normalized ICF...")
    try:
        word_icf = compute_normalized_icf(word_counts, total_tokens)
    except Exception as e:
        print(f"Error computing ICF: {e}")
        raise

    # Stratified sampling
    print("Creating stratified sample...")
    try:
        samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
        print(f"Sampled {len(samples)} word-ICF pairs")
    except Exception as e:
        print(f"Error in stratified sampling: {e}")
        raise

    # Split train/val (80/20)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Datasets
    train_dataset = WordICFDataset(
        train_samples, max_length=args.max_length, augment_prob=args.augment_prob
    )
    val_dataset = WordICFDataset(val_samples, max_length=args.max_length, augment_prob=0.0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model with proper initialization
    try:
        model = UniversalICF().to(device)
        sample_icf_values = [icf for _, icf in samples[:1000]]
        mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
        model.init_weights(mean_icf=mean_icf)
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Initialized with mean ICF bias: {mean_icf:.4f}")
    except Exception as e:
        print(f"Error creating model: {e}")
        raise

    # Loss and optimizer
    if args.curriculum:
        base_loss = EnhancedMultiLoss()
        criterion = CurriculumMultiLoss(args.epochs, 0, base_loss)
    elif args.multi_loss:
        criterion = EnhancedMultiLoss()
    else:
        criterion = CombinedLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Update curriculum stage if using curriculum loss
        if args.curriculum and isinstance(criterion, CurriculumMultiLoss):
            criterion.current_epoch = epoch
            criterion.base_loss = criterion.base_loss  # Trigger stage update

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_multi_loss=args.multi_loss or args.curriculum,
        )
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                # Ensure output directory exists
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.output)
                print(f"Saved best model to {args.output}")
            except Exception as e:
                print(f"Error saving model: {e}")
                raise

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
