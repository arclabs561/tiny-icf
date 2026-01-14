#!/usr/bin/env -S uv run
"""Deep analysis of training dynamics to understand low correlation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_icf.model import UniversalICF
from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.loss import CombinedLoss
from tiny_icf.train import generate_ranking_pairs, set_seed
from tiny_icf.eval import compute_metrics


def analyze_training_dynamics():
    """Deep dive into what's happening during training."""
    set_seed(42)
    device = torch.device("cpu")
    
    # Load small subset for analysis
    data_path = Path("data/word_frequency.csv")
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    test_samples = samples[:2000]  # Small for quick analysis
    split_idx = int(len(test_samples) * 0.8)
    train_samples = test_samples[:split_idx]
    val_samples = test_samples[split_idx:]
    
    train_dataset = WordICFDataset(train_samples, max_length=20, augment_prob=0.1)
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print("=" * 70)
    print("Deep Training Dynamics Analysis")
    print("=" * 70)
    print(f"Training on {len(train_samples)} words, {len(val_samples)} validation\n")
    
    # Model
    model = UniversalICF().to(device)
    sample_icf_values = [icf for _, icf in train_samples[:1000]]
    mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
    model.init_weights(mean_icf=mean_icf)
    
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Track detailed metrics
    epoch_metrics = []
    
    for epoch in range(10):
        model.train()
        epoch_losses = []
        epoch_huber = []
        epoch_ranking = []
        all_preds = []
        all_targets = []
        
        for byte_tensors, icf_targets in train_loader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(byte_tensors)
            
            # Generate pairs
            n_pairs = min(len(icf_targets), 32)
            pairs = generate_ranking_pairs(icf_targets, n_pairs)
            
            # Compute loss components
            from tiny_icf.loss import huber_loss, ranking_loss
            huber = huber_loss(predictions, icf_targets, delta=0.1)
            
            rank = torch.tensor(0.0)
            if len(pairs) > 0:
                idx1, idx2 = pairs[:, 0], pairs[:, 1]
                rank = ranking_loss(predictions[idx1], predictions[idx2], margin=0.1)
            
            total_loss = huber + 2.0 * rank
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            epoch_huber.append(huber.item())
            epoch_ranking.append(rank.item())
            all_preds.append(predictions.detach().cpu().numpy())
            all_targets.append(icf_targets.detach().cpu().numpy())
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for byte_tensors, icf_targets in val_loader:
                byte_tensors = byte_tensors.to(device)
                icf_targets = icf_targets.to(device)
                predictions = model(byte_tensors)
                val_preds.append(predictions.cpu().numpy())
                val_targets.append(icf_targets.cpu().numpy())
        
        train_preds = np.concatenate(all_preds)
        train_targets = np.concatenate(all_targets)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        train_metrics = compute_metrics(train_preds, train_targets)
        val_metrics = compute_metrics(val_preds, val_targets)
        
        print(f"\nEpoch {epoch+1}/10:")
        print(f"  Loss: {np.mean(epoch_losses):.4f} (Huber: {np.mean(epoch_huber):.4f}, Rank: {np.mean(epoch_ranking):.4f})")
        print(f"  Train Spearman: {train_metrics['spearman_corr']:.4f}")
        print(f"  Val Spearman: {val_metrics['spearman_corr']:.4f}")
        print(f"  Train Pred Range: [{train_preds.min():.4f}, {train_preds.max():.4f}] (std: {train_preds.std():.4f})")
        print(f"  Val Pred Range: [{val_preds.min():.4f}, {val_preds.max():.4f}] (std: {val_preds.std():.4f})")
        
        # Analyze ranking pairs
        if len(pairs) > 0:
            idx1, idx2 = pairs[:, 0], pairs[:, 1]
            pred1 = predictions[idx1].detach().cpu().numpy()
            pred2 = predictions[idx2].detach().cpu().numpy()
            target1 = icf_targets[idx1].detach().cpu().numpy()
            target2 = icf_targets[idx2].detach().cpu().numpy()
            
            # Check if ranking is correct
            correct_rankings = ((target1 < target2) == (pred1 < pred2)).sum()
            ranking_accuracy = correct_rankings / len(pairs) if len(pairs) > 0 else 0.0
            print(f"  Ranking Accuracy: {ranking_accuracy:.2%} ({correct_rankings}/{len(pairs)})")
        
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_spearman': train_metrics['spearman_corr'],
            'val_spearman': val_metrics['spearman_corr'],
            'train_std': train_preds.std(),
            'val_std': val_preds.std(),
            'ranking_loss': np.mean(epoch_ranking),
        })
    
    # Summary analysis
    print("\n" + "=" * 70)
    print("Summary Analysis")
    print("=" * 70)
    
    final_spearman = epoch_metrics[-1]['val_spearman']
    initial_spearman = epoch_metrics[0]['val_spearman']
    
    print(f"Spearman Correlation:")
    print(f"  Initial: {initial_spearman:.4f}")
    print(f"  Final: {final_spearman:.4f}")
    print(f"  Change: {final_spearman - initial_spearman:+.4f}")
    
    if final_spearman < 0.3:
        print("\n⚠ Low correlation detected. Possible causes:")
        print("  1. Model not learning ranking relationships")
        print("  2. Ranking loss too weak or not contributing")
        print("  3. Batch size too small for effective ranking pairs")
        print("  4. Learning rate too high/low")
        print("  5. Model capacity insufficient")
    
    # Check ranking loss contribution
    avg_ranking_loss = np.mean([m['ranking_loss'] for m in epoch_metrics])
    if avg_ranking_loss < 0.01:
        print(f"\n⚠ Ranking loss very small ({avg_ranking_loss:.6f})")
        print("  Consider increasing rank_weight or rank_margin")
    
    # Check prediction range expansion
    initial_std = epoch_metrics[0]['val_std']
    final_std = epoch_metrics[-1]['val_std']
    print(f"\nPrediction Range Expansion:")
    print(f"  Initial std: {initial_std:.4f}")
    print(f"  Final std: {final_std:.4f}")
    print(f"  Expansion: {final_std / initial_std:.2f}x")
    
    if final_std < 0.05:
        print("  ⚠ Predictions still compressed - may need stronger ranking signal")


if __name__ == "__main__":
    analyze_training_dynamics()

