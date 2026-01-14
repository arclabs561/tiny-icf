#!/usr/bin/env -S uv run
"""Quick test of improvements on small subset."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.eval import compute_metrics, evaluate_jabberwocky
from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.train import generate_ranking_pairs, set_seed, train_epoch, validate


def quick_test():
    """Test improvements on small dataset."""
    from tiny_icf.train import set_seed
    
    set_seed(42)
    device = torch.device("cpu")
    
    # Load small subset
    data_path = Path("data/word_frequency.csv")
    if not data_path.exists():
        print("Data file not found, skipping test")
        return
    
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    # Use small subset for quick test
    test_samples = samples[:5000]  # 5k words
    split_idx = int(len(test_samples) * 0.8)
    train_samples = test_samples[:split_idx]
    val_samples = test_samples[split_idx:]
    
    train_dataset = WordICFDataset(train_samples, max_length=20, augment_prob=0.1)
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model with improved initialization
    model = UniversalICF().to(device)
    sample_icf_values = [icf for _, icf in train_samples[:1000]]
    mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
    model.init_weights(mean_icf=mean_icf)
    
    print(f"Testing improvements on {len(test_samples)} words")
    print(f"Initialized with mean ICF: {mean_icf:.4f}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Enhanced loss
    criterion = CombinedLoss()  # Now with rank_weight=2.0, rank_margin=0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for 5 epochs
    print("\nTraining for 5 epochs...")
    for epoch in range(5):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/5: Train loss = {train_loss:.4f}")
        
        # Quick evaluation
        if (epoch + 1) % 2 == 0:
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for byte_tensors, icf_targets in val_loader:
                    byte_tensors = byte_tensors.to(device)
                    icf_targets = icf_targets.to(device)
                    predictions = model(byte_tensors)
                    all_preds.append(predictions.cpu().numpy())
                    all_targets.append(icf_targets.cpu().numpy())
            
            import numpy as np
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
            metrics = compute_metrics(preds, targets)
            
            print(f"  Eval: MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_corr']:.4f}")
            print(f"  Predictions: mean={preds.mean():.4f}, std={preds.std():.4f}, range=[{preds.min():.4f}, {preds.max():.4f}]")
            model.train()
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for byte_tensors, icf_targets in val_loader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            predictions = model(byte_tensors)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(icf_targets.cpu().numpy())
    
    import numpy as np
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(preds, targets)
    jabberwocky = evaluate_jabberwocky(model, device)
    
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Spearman: {metrics['spearman_corr']:.4f}")
    print(f"Jabberwocky: {jabberwocky['passed_count']}/{jabberwocky['total_count']} ({jabberwocky['pass_rate']:.1%})")
    print(f"Predictions: mean={preds.mean():.4f}, std={preds.std():.4f}, range=[{preds.min():.4f}, {preds.max():.4f}]")
    print(f"Targets: mean={targets.mean():.4f}, std={targets.std():.4f}, range=[{targets.min():.4f}, {targets.max():.4f}]")
    
    # Compare to baseline
    print("\n=== Improvement Check ===")
    pred_range = preds.max() - preds.min()
    target_range = targets.max() - targets.min()
    range_ratio = pred_range / target_range if target_range > 0 else 0
    
    print(f"Prediction range ratio: {range_ratio:.2%} (target: >80%)")
    print(f"Prediction std: {preds.std():.4f} (target: >0.05)")
    
    if preds.std() > 0.03 and metrics['spearman_corr'] > 0.3:
        print("✓ Improvements show promise!")
    else:
        print("⚠ May need further tuning")


if __name__ == "__main__":
    quick_test()

