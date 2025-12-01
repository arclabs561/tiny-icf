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


def generate_ranking_pairs(targets: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """
    Generate pairs for ranking loss where target[i] < target[j] (i more common).
    
    Args:
        targets: [Batch, 1] or [Batch] ground truth ICF scores
        n_pairs: Number of pairs to generate
    
    Returns:
        [N_pairs, 2] tensor of indices (i, j) where target[i] < target[j]
    """
    batch_size = len(targets)
    if batch_size < 2:
        return torch.empty((0, 2), dtype=torch.long, device=targets.device)
    
    pairs = []
    # Handle both [Batch, 1] and [Batch] shapes
    if targets.dim() > 1 and targets.size(1) == 1:
        targets_flat = targets.squeeze(1)  # [Batch]
    else:
        targets_flat = targets  # Already [Batch]
    
    for _ in range(n_pairs):
        i, j = torch.randint(0, batch_size, (2,), device=targets.device)
        if i == j:
            continue
        
        # Ensure i is more common (lower ICF) than j
        if targets_flat[i] < targets_flat[j]:
            pairs.append([i.item(), j.item()])
        elif targets_flat[j] < targets_flat[i]:
            pairs.append([j.item(), i.item()])
    
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=targets.device)
    
    return torch.tensor(pairs, dtype=torch.long, device=targets.device)


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
    
    for byte_tensors, icf_targets in tqdm(dataloader, desc="Training"):
        byte_tensors = byte_tensors.to(device)
        icf_targets = icf_targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(byte_tensors)
        
        # Generate pairs for ranking loss
        n_pairs = len(icf_targets) // 2
        pairs = generate_ranking_pairs(icf_targets, n_pairs)
        
        # Compute loss with ranking pairs
        loss = criterion(predictions, icf_targets, pairs=pairs)
        loss.backward()
        
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


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
        print(f"Model parameters: {model.count_parameters():,}")
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

