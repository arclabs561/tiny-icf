#!/usr/bin/env -S uv run
"""
Quick test of listwise loss training on small dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.loss import CombinedLoss  # TODO: Review - migrated from CombinedListwiseLoss
from tiny_icf.model import UniversalICF
from tiny_icf.eval import compute_metrics
from tiny_icf.eval_rbo import compute_rbo_metrics


def main():
    print("="*70)
    print("Quick Test: Listwise Loss Training")
    print("="*70)
    
    # Load small subset
    data_path = Path("data/word_frequency.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return
    
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    # Use small subset for quick test
    train_samples = samples[:2000]
    val_samples = samples[2000:2500]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Datasets
    train_dataset = WordICFDataset(train_samples, max_length=20, augment_prob=0.1)
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model
    device = torch.device("cpu")
    model = UniversalICF().to(device)
    mean_icf = sum([icf for _, icf in train_samples[:100]]) / min(100, len(train_samples))
    model.init_weights(mean_icf=mean_icf)
    
    # Loss
    criterion = CombinedLoss  # TODO: Review - migrated from CombinedListwiseLoss(
        huber_delta=0.1,
        listwise_weight=1.0,
        listwise_method="lambdarank",
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTraining for 5 epochs...")
    for epoch in range(1, 6):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for byte_tensors, icf_targets in train_loader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(byte_tensors)
            
            # Collapse detection
            if predictions.std().item() < 0.01:
                print(f"  ⚠ Collapse detected!")
                break
            
            loss = criterion(predictions, icf_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Validate
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for byte_tensors, icf_targets in val_loader:
                byte_tensors = byte_tensors.to(device)
                icf_targets = icf_targets.to(device)
                predictions = model(byte_tensors)
                all_preds.append(predictions.cpu())
                all_targets.append(icf_targets.cpu())
        
        predictions = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        
        metrics = compute_metrics(predictions, targets)
        
        # Add RBO
        try:
            rbo_metrics = compute_rbo_metrics(
                torch.tensor(predictions),
                torch.tensor(targets),
            )
            metrics.update(rbo_metrics)
        except Exception:
            pass
        
        print(f"Epoch {epoch}/5: loss={total_loss/n_batches:.4f}, "
              f"MAE={metrics['mae']:.4f}, Spearman={metrics.get('spearman_corr', 0.0):.4f}, "
              f"RBO={metrics.get('rbo_full', 0.0):.4f}")
    
    print("\n✅ Quick test complete!")


if __name__ == "__main__":
    main()

