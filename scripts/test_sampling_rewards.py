#!/usr/bin/env -S uv run
"""Compare training with and without sampling-based rewards."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.eval import compute_metrics
from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.train import generate_ranking_pairs, set_seed


def train_with_sampling(model, dataloader, criterion, optimizer, device, use_weighted=True):
    """Train one epoch with optional weighted sampling."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for byte_tensors, icf_targets in tqdm(dataloader, desc=f"Training (weighted={use_weighted})", leave=False):
        byte_tensors = byte_tensors.to(device)
        icf_targets = icf_targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(byte_tensors)
        
        n_pairs = min(len(icf_targets), 32)
        pairs, pair_diffs = generate_ranking_pairs(
            icf_targets, n_pairs, min_diff=0.05, use_weighted_sampling=use_weighted
        )
        
        loss = criterion(
            predictions, icf_targets,
            pairs=pairs,
            pair_target_diffs=pair_diffs if use_weighted else None,
            smooth_ranking=True,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for byte_tensors, icf_targets in dataloader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            predictions = model(byte_tensors)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(icf_targets.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(preds, targets)
    
    return metrics, preds, targets


def main():
    """Compare weighted vs uniform sampling."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load small subset for quick comparison
    data_path = Path("data/word_frequency.csv")
    if not data_path.exists():
        print("Data file not found, skipping test")
        return
    
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    # Use small subset
    test_samples = samples[:5000]
    split_idx = int(len(test_samples) * 0.8)
    train_samples = test_samples[:split_idx]
    val_samples = test_samples[split_idx:]
    
    train_dataset = WordICFDataset(train_samples, max_length=20, augment_prob=0.1)
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print("=" * 70)
    print("Comparing Weighted vs Uniform Sampling")
    print("=" * 70)
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print()
    
    results = {}
    
    for use_weighted in [False, True]:
        name = "Weighted" if use_weighted else "Uniform"
        print(f"\n{'='*70}")
        print(f"Training with {name} Sampling")
        print(f"{'='*70}\n")
        
        # Create fresh model
        model = UniversalICF().to(device)
        sample_icf_values = [icf for _, icf in train_samples[:1000]]
        mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
        model.init_weights(mean_icf=mean_icf)
        
        criterion = CombinedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for 5 epochs
        for epoch in range(5):
            train_loss = train_with_sampling(
                model, train_loader, criterion, optimizer, device, use_weighted=use_weighted
            )
            if (epoch + 1) % 2 == 0:
                metrics, preds, targets = evaluate(model, val_loader, device)
                print(f"Epoch {epoch+1}/5: Loss={train_loss:.4f}, "
                      f"MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_corr']:.4f}, "
                      f"std={preds.std():.4f}")
        
        # Final evaluation
        metrics, preds, targets = evaluate(model, val_loader, device)
        results[name] = {
            'metrics': metrics,
            'preds': preds,
            'targets': targets,
        }
        
        print(f"\nFinal {name} Results:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Spearman: {metrics['spearman_corr']:.4f}")
        print(f"  Predictions: mean={preds.mean():.4f}, std={preds.std():.4f}, "
              f"range=[{preds.min():.4f}, {preds.max():.4f}]")
    
    # Compare results
    print(f"\n{'='*70}")
    print("Comparison Summary")
    print(f"{'='*70}\n")
    
    uniform = results['Uniform']
    weighted = results['Weighted']
    
    print(f"Spearman Correlation:")
    print(f"  Uniform:  {uniform['metrics']['spearman_corr']:.4f}")
    print(f"  Weighted: {weighted['metrics']['spearman_corr']:.4f}")
    print(f"  Improvement: {weighted['metrics']['spearman_corr'] - uniform['metrics']['spearman_corr']:+.4f}")
    print()
    
    print(f"Prediction Standard Deviation:")
    print(f"  Uniform:  {uniform['preds'].std():.4f}")
    print(f"  Weighted: {weighted['preds'].std():.4f}")
    print(f"  Improvement: {weighted['preds'].std() - uniform['preds'].std():+.4f}")
    print()
    
    print(f"MAE:")
    print(f"  Uniform:  {uniform['metrics']['mae']:.4f}")
    print(f"  Weighted: {weighted['metrics']['mae']:.4f}")
    print(f"  Improvement: {uniform['metrics']['mae'] - weighted['metrics']['mae']:+.4f}")
    
    if weighted['metrics']['spearman_corr'] > uniform['metrics']['spearman_corr']:
        print("\n✓ Weighted sampling shows improvement!")
    else:
        print("\n⚠ Weighted sampling needs tuning")


if __name__ == "__main__":
    main()

