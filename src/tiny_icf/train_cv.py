"""Training script with cross-validation support."""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.train import generate_ranking_pairs


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# generate_ranking_pairs imported from train.py


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch with ranking loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for byte_tensors, icf_targets in tqdm(dataloader, desc="Training", leave=False):
        byte_tensors = byte_tensors.to(device)
        icf_targets = icf_targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(byte_tensors)
        
        # Generate pairs for ranking loss with weighted sampling
        n_pairs = min(len(icf_targets), 32)  # More pairs for better ranking learning
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
        for byte_tensors, icf_targets in tqdm(dataloader, desc="Validating", leave=False):
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            predictions = model(byte_tensors)
            loss = criterion(predictions, icf_targets)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def train_fold(
    train_indices: List[int],
    val_indices: List[int],
    full_dataset: WordICFDataset,
    fold_num: int,
    args,
    device: torch.device,
) -> Tuple[float, str]:
    """Train a single fold."""
    # Create subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    
    # Create fresh model for this fold
    try:
        model = UniversalICF().to(device)
        criterion = CombinedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    except Exception as e:
        print(f"Error creating model for fold {fold_num}: {e}")
        raise
    
    best_val_loss = float("inf")
    best_model_path = args.output.parent / f"{args.output.stem}_fold{fold_num}.pt"
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
            except Exception as e:
                print(f"Error saving model for fold {fold_num}: {e}")
                raise
    
    return best_val_loss, str(best_model_path)


def main():
    parser = argparse.ArgumentParser(description="Train Universal ICF model with cross-validation")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=20, help="Max word length")
    parser.add_argument("--augment-prob", type=float, default=0.1, help="Augmentation probability")
    parser.add_argument("--output", type=Path, default=Path("model_cv.pt"), help="Output model path")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
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
    print(f"Cross-validation with {args.folds} folds")
    
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
    
    # Create full dataset
    full_dataset = WordICFDataset(samples, max_length=args.max_length, augment_prob=args.augment_prob)
    
    # Cross-validation
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results = []
    
    print(f"\n{'='*80}")
    print(f"Starting {args.folds}-fold cross-validation")
    print(f"{'='*80}\n")
    
    for fold_num, (train_indices, val_indices) in enumerate(kfold.split(samples), 1):
        print(f"\nFold {fold_num}/{args.folds}")
        print(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
        
        val_loss, model_path = train_fold(
            train_indices.tolist(),
            val_indices.tolist(),
            full_dataset,
            fold_num,
            args,
            device,
        )
        
        fold_results.append((fold_num, val_loss, model_path))
        print(f"Fold {fold_num} - Best Val Loss: {val_loss:.4f}")
        print(f"  Model saved to: {model_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("Cross-Validation Results")
    print(f"{'='*80}")
    print(f"{'Fold':<10} {'Val Loss':<15} {'Model Path'}")
    print("-" * 80)
    
    val_losses = []
    for fold_num, val_loss, model_path in fold_results:
        print(f"{fold_num:<10} {val_loss:<15.4f} {model_path}")
        val_losses.append(val_loss)
    
    mean_loss = sum(val_losses) / len(val_losses)
    std_loss = (sum((x - mean_loss) ** 2 for x in val_losses) / len(val_losses)) ** 0.5
    
    print("-" * 80)
    print(f"Mean Val Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    
    # Find best fold
    best_fold = min(fold_results, key=lambda x: x[1])
    print(f"\nBest fold: {best_fold[0]} (Val Loss: {best_fold[1]:.4f})")
    print(f"Best model: {best_fold[2]}")
    
    # Optionally copy best model to main output path
    import shutil
    shutil.copy(best_fold[2], args.output)
    print(f"Copied best model to: {args.output}")


if __name__ == "__main__":
    main()

