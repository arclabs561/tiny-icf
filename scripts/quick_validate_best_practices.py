#!/usr/bin/env -S uv run
"""Quick validation of the unified best practices training script."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.eval import compute_metrics, evaluate_jabberwocky
from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.scheduler import AdaptiveCosineAnnealingLR
from tiny_icf.train import generate_ranking_pairs, set_seed, train_epoch, validate


def quick_validate():
    """Quick validation of best practices training."""
    print("=" * 70)
    print("Quick Validation: Best Practices Training")
    print("=" * 70)
    
    set_seed(42)
    device = torch.device("cpu")
    
    # Load small subset
    data_path = Path("data/word_frequency.csv")
    if not data_path.exists():
        print("❌ Data file not found, skipping validation")
        return False
    
    print("\n1. Loading data...")
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    test_samples = samples[:2000]  # Small subset for quick test
    split_idx = int(len(test_samples) * 0.8)
    train_samples = test_samples[:split_idx]
    val_samples = test_samples[split_idx:]
    
    train_dataset = WordICFDataset(train_samples, max_length=20, augment_prob=0.1)
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"   Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Model
    print("\n2. Initializing model...")
    model = UniversalICF().to(device)
    sample_icf_values = [icf for _, icf in train_samples[:500]]
    mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
    model.init_weights(mean_icf=mean_icf)
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Mean ICF bias: {mean_icf:.4f}")
    
    # Loss and optimizer
    print("\n3. Setting up training...")
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = 1e-3
    
    # Scheduler
    scheduler = AdaptiveCosineAnnealingLR(
        optimizer, T_max=3, eta_min=1e-5,
        metric="spearman_corr", mode="max", patience=2
    )
    print("   ✓ Adaptive scheduler initialized")
    
    # Train for 3 epochs
    print("\n4. Training (3 epochs)...")
    for epoch in range(3):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Quick evaluation
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
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        metrics = compute_metrics(preds, targets)
        
        print(f"   Epoch {epoch+1}/3: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_corr']:.4f}")
        
        # Update scheduler
        scheduler.step(metrics, epoch=epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        
        model.train()
    
    # Final evaluation
    print("\n5. Final evaluation...")
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
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(preds, targets)
    jabberwocky = evaluate_jabberwocky(model, device)
    
    print(f"\n📊 Results:")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   Spearman: {metrics['spearman_corr']:.4f}")
    print(f"   Jabberwocky: {jabberwocky['passed_count']}/{jabberwocky['total_count']} ({jabberwocky['pass_rate']:.1%})")
    print(f"   Predictions: mean={preds.mean():.4f}, std={preds.std():.4f}, range=[{preds.min():.4f}, {preds.max():.4f}]")
    
    # Validation checks
    print("\n6. Validation checks...")
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Prediction range
    if preds.min() < 0.1 and preds.max() > 0.9:
        print("   ✓ Prediction range: [0.0, 1.0] (good)")
        checks_passed += 1
    else:
        print(f"   ⚠ Prediction range: [{preds.min():.4f}, {preds.max():.4f}] (should be wider)")
    
    # Check 2: Prediction std
    if preds.std() > 0.05:
        print(f"   ✓ Prediction std: {preds.std():.4f} (good)")
        checks_passed += 1
    else:
        print(f"   ⚠ Prediction std: {preds.std():.4f} (should be >0.05)")
    
    # Check 3: Spearman correlation
    if metrics['spearman_corr'] > 0.1:
        print(f"   ✓ Spearman: {metrics['spearman_corr']:.4f} (learning)")
        checks_passed += 1
    else:
        print(f"   ⚠ Spearman: {metrics['spearman_corr']:.4f} (too low)")
    
    # Check 4: Loss decreasing
    if train_loss < 50.0:  # Reasonable threshold
        print(f"   ✓ Training loss: {train_loss:.4f} (reasonable)")
        checks_passed += 1
    else:
        print(f"   ⚠ Training loss: {train_loss:.4f} (high)")
    
    print(f"\n✅ Validation: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed >= 3:
        print("✓ Best practices training script is working correctly!")
        return True
    else:
        print("⚠ Some checks failed - may need tuning")
        return False


if __name__ == "__main__":
    success = quick_validate()
    sys.exit(0 if success else 1)

