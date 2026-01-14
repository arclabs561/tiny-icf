#!/usr/bin/env -S uv run
"""Compare different loss configurations to find optimal settings."""

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


def train_with_config(model, dataloader, criterion, optimizer, device, config_name):
    """Train one epoch with a specific configuration."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for byte_tensors, icf_targets in tqdm(dataloader, desc=f"Training ({config_name})", leave=False):
        byte_tensors = byte_tensors.to(device)
        icf_targets = icf_targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(byte_tensors)
        
        n_pairs = min(len(icf_targets), 32)
        pairs, pair_diffs = generate_ranking_pairs(
            icf_targets, n_pairs, min_diff=0.05, use_weighted_sampling=True
        )
        
        loss = criterion(
            predictions, icf_targets,
            pairs=pairs,
            pair_target_diffs=pair_diffs,
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
    """Compare different loss configurations."""
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
    
    # Test different loss configurations
    configs = [
        {"rank_weight": 2.0, "rank_margin": 0.1, "name": "Baseline (2.0, 0.1)"},
        {"rank_weight": 3.0, "rank_margin": 0.1, "name": "Stronger ranking (3.0, 0.1)"},
        {"rank_weight": 2.0, "rank_margin": 0.15, "name": "Larger margin (2.0, 0.15)"},
        {"rank_weight": 4.0, "rank_margin": 0.1, "name": "Very strong (4.0, 0.1)"},
        {"rank_weight": 1.5, "rank_margin": 0.2, "name": "Wide margin (1.5, 0.2)"},
    ]
    
    print("=" * 70)
    print("Comparing Loss Configurations")
    print("=" * 70)
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print()
    
    results = {}
    
    for config in configs:
        name = config["name"]
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}\n")
        
        # Create fresh model
        model = UniversalICF().to(device)
        sample_icf_values = [icf for _, icf in train_samples[:1000]]
        mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
        model.init_weights(mean_icf=mean_icf)
        
        criterion = CombinedLoss(
            rank_weight=config["rank_weight"],
            rank_margin=config["rank_margin"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for 5 epochs
        for epoch in range(5):
            train_loss = train_with_config(
                model, train_loader, criterion, optimizer, device, name
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
            'config': config,
        }
        
        print(f"\nFinal {name} Results:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Spearman: {metrics['spearman_corr']:.4f}")
        print(f"  Predictions: mean={preds.mean():.4f}, std={preds.std():.4f}, "
              f"range=[{preds.min():.4f}, {preds.max():.4f}]")
    
    # Compare results
    print(f"\n{'='*70}")
    print("Configuration Comparison")
    print(f"{'='*70}\n")
    
    print(f"{'Config':<30} {'MAE':<8} {'Spearman':<10} {'Std':<8} {'Range':<15}")
    print("-" * 70)
    
    best_spearman = -1
    best_config = None
    
    for name, result in results.items():
        metrics = result['metrics']
        preds = result['preds']
        pred_range = f"[{preds.min():.3f}, {preds.max():.3f}]"
        
        print(f"{name:<30} {metrics['mae']:<8.4f} {metrics['spearman_corr']:<10.4f} "
              f"{preds.std():<8.4f} {pred_range:<15}")
        
        if metrics['spearman_corr'] > best_spearman:
            best_spearman = metrics['spearman_corr']
            best_config = name
    
    print(f"\n✓ Best configuration: {best_config} (Spearman: {best_spearman:.4f})")
    print(f"  Rank weight: {results[best_config]['config']['rank_weight']}")
    print(f"  Rank margin: {results[best_config]['config']['rank_margin']}")


if __name__ == "__main__":
    main()

