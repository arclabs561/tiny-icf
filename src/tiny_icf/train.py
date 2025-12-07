"""Training script for Universal ICF model."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_ranking_pairs(
    targets: torch.Tensor, 
    n_pairs: int, 
    min_diff: float = 0.05,
    use_weighted_sampling: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate pairs for ranking loss with sampling-based rewards.
    
    Uses weighted sampling: pairs with larger ICF differences are sampled
    with higher probability, providing stronger learning signal.
    
    Args:
        targets: [Batch, 1] or [Batch] ground truth ICF scores
        n_pairs: Number of pairs to generate
        min_diff: Minimum ICF difference required (default: 0.05)
        use_weighted_sampling: If True, sample pairs weighted by ICF difference
    
    Returns:
        (pairs, diffs) where:
        - pairs: [N_pairs, 2] tensor of indices (i, j) where target[i] < target[j]
        - diffs: [N_pairs] tensor of actual ICF differences for weighted loss
    """
    batch_size = len(targets)
    if batch_size < 2:
        empty_pairs = torch.empty((0, 2), dtype=torch.long, device=targets.device)
        empty_diffs = torch.empty((0,), dtype=targets.dtype, device=targets.device)
        return empty_pairs, empty_diffs
    
    # Handle both [Batch, 1] and [Batch] shapes
    if targets.dim() > 1 and targets.size(1) == 1:
        targets_flat = targets.squeeze(1)  # [Batch]
    else:
        targets_flat = targets  # Already [Batch]
    
    # Build all valid pairs with their differences
    valid_pairs = []
    valid_diffs = []
    
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if targets_flat[i] < targets_flat[j]:
                diff = targets_flat[j] - targets_flat[i]
                if diff >= min_diff:
                    valid_pairs.append([i, j])
                    valid_diffs.append(diff.item())
            elif targets_flat[j] < targets_flat[i]:
                diff = targets_flat[i] - targets_flat[j]
                if diff >= min_diff:
                    valid_pairs.append([j, i])
                    valid_diffs.append(diff.item())
    
    if not valid_pairs:
        # Fallback: use pairs without min_diff requirement
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if targets_flat[i] < targets_flat[j]:
                    valid_pairs.append([i, j])
                    valid_diffs.append((targets_flat[j] - targets_flat[i]).item())
                elif targets_flat[j] < targets_flat[i]:
                    valid_pairs.append([j, i])
                    valid_diffs.append((targets_flat[i] - targets_flat[j]).item())
    
    if not valid_pairs:
        empty_pairs = torch.empty((0, 2), dtype=torch.long, device=targets.device)
        empty_diffs = torch.empty((0,), dtype=targets.dtype, device=targets.device)
        return empty_pairs, empty_diffs
    
    # Convert to tensors
    valid_pairs_tensor = torch.tensor(valid_pairs, dtype=torch.long, device=targets.device)
    valid_diffs_tensor = torch.tensor(valid_diffs, dtype=targets.dtype, device=targets.device)
    
    # Sample pairs
    if use_weighted_sampling and len(valid_pairs) > n_pairs:
        # Weighted sampling: probability proportional to ICF difference
        # Use softmax to create probability distribution
        # Higher temperature = more uniform, lower = more focused on large diffs
        probs = torch.softmax(valid_diffs_tensor * 5.0, dim=0)  # Scale factor controls emphasis
        
        # Sample n_pairs indices according to probabilities
        indices = torch.multinomial(probs, num_samples=min(n_pairs, len(valid_pairs)), replacement=False)
        pairs = valid_pairs_tensor[indices]
        diffs = valid_diffs_tensor[indices]
    else:
        # Uniform sampling or use all pairs if fewer than n_pairs
        if len(valid_pairs) > n_pairs:
            indices = torch.randperm(len(valid_pairs), device=targets.device)[:n_pairs]
            pairs = valid_pairs_tensor[indices]
            diffs = valid_diffs_tensor[indices]
        else:
            pairs = valid_pairs_tensor
            diffs = valid_diffs_tensor
    
    return pairs, diffs


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_loss_components: bool = False,
    check_collapse: bool = True,
) -> tuple[float, dict]:
    """
    Train for one epoch with ranking loss.
    
    Returns:
        (average_loss, metrics_dict) where metrics_dict contains:
        - huber_loss: Average Huber loss
        - ranking_loss: Average ranking loss (if pairs provided)
        - pred_std: Standard deviation of predictions (for collapse detection)
        - pred_range: [min, max] of predictions
    """
    model.train()
    total_loss = 0.0
    total_huber = 0.0
    total_ranking = 0.0
    n_batches = 0
    all_predictions = []
    
    for byte_tensors, icf_targets in tqdm(dataloader, desc="Training"):
        byte_tensors = byte_tensors.to(device)
        icf_targets = icf_targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(byte_tensors)
        
        # Collapse detection: check prediction variance
        if check_collapse:
            pred_std = predictions.std().item()
            if pred_std < 0.01:
                raise RuntimeError(
                    f"Model collapsed: prediction std={pred_std:.6f} < 0.01. "
                    "All predictions are too similar. Check model initialization and loss function."
                )
        
        # Collect predictions for analysis
        all_predictions.append(predictions.detach().cpu())
        
        # Generate pairs for ranking loss with weighted sampling
        n_pairs = min(len(icf_targets), 32)
        pairs, pair_diffs = generate_ranking_pairs(
            icf_targets, n_pairs, min_diff=0.05, use_weighted_sampling=True
        )
        
        # Compute loss with ranking pairs and smooth rewards
        loss = criterion(
            predictions, icf_targets, 
            pairs=pairs, 
            pair_target_diffs=pair_diffs,
            smooth_ranking=True,
        )
        
        # Extract loss components if logging enabled
        if log_loss_components:
            # Compute Huber loss separately
            from tiny_icf.loss import huber_loss
            huber = huber_loss(predictions, icf_targets, delta=0.1)
            total_huber += huber.item()
            
            # Compute ranking loss separately if pairs exist
            if len(pairs) > 0:
                from tiny_icf.loss import ranking_loss
                idx1, idx2 = pairs[:, 0], pairs[:, 1]
                rank = ranking_loss(
                    predictions[idx1], predictions[idx2],
                    margin=0.1,
                    target_diff=pair_diffs,
                    smooth=True,
                )
                total_ranking += rank.item()
        
        loss.backward()
        
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    # Compute metrics
    all_preds = torch.cat(all_predictions, dim=0)
    metrics = {
        "pred_std": all_preds.std().item(),
        "pred_min": all_preds.min().item(),
        "pred_max": all_preds.max().item(),
        "pred_mean": all_preds.mean().item(),
    }
    
    if log_loss_components:
        metrics["huber_loss"] = total_huber / n_batches if n_batches > 0 else 0.0
        metrics["ranking_loss"] = total_ranking / n_batches if n_batches > 0 else 0.0
    
    return total_loss / n_batches if n_batches > 0 else 0.0, metrics


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
            loss = criterion(predictions, icf_targets)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train Universal ICF model")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=20, help="Max word length")
    parser.add_argument("--augment-prob", type=float, default=0.1, help="Augmentation probability")
    parser.add_argument("--output", type=Path, default=Path("model.pt"), help="Output model path")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Random seed: 42 (reproducible)")
    
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
    train_dataset = WordICFDataset(train_samples, max_length=args.max_length, augment_prob=args.augment_prob)
    val_dataset = WordICFDataset(val_samples, max_length=args.max_length, augment_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    try:
        model = UniversalICF().to(device)
        # Initialize weights properly
        # Estimate mean ICF from data for better initialization
        sample_icf_values = [icf for _, icf in samples[:1000]]
        mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
        model.init_weights(mean_icf=mean_icf)
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Initialized with mean ICF bias: {mean_icf:.4f}")
    except Exception as e:
        print(f"Error creating model: {e}")
        raise
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
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

