#!/usr/bin/env -S uv run
"""Quick test of multi-loss training on small subset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_icf.model import UniversalICF
from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.loss import CombinedLoss
from tiny_icf.loss import CombinedLoss  # TODO: Review - migrated from EnhancedMultiLoss
from tiny_icf.train import generate_ranking_pairs, set_seed, validate
from tiny_icf.train_multi_loss import get_common_rare_indices


def quick_test_multi_loss():
    """Test multi-loss vs standard loss on small subset."""
    set_seed(42)
    device = torch.device("cpu")
    
    # Load small subset
    data_path = Path("data/word_frequency.csv")
    if not data_path.exists():
        print("Data file not found")
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("=" * 70)
    print("Multi-Loss vs Standard Loss Comparison")
    print("=" * 70)
    print(f"Training on {len(train_samples)} words, {len(val_samples)} validation")
    print()
    
    results = {}
    
    # Test 1: Standard CombinedLoss
    print("1. Training with Standard CombinedLoss...")
    model1 = UniversalICF().to(device)
    sample_icf_values = [icf for _, icf in train_samples[:1000]]
    mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
    model1.init_weights(mean_icf=mean_icf)
    
    criterion1 = CombinedLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    
    for epoch in range(5):
        model1.train()
        train_loss = 0.0
        n_batches = 0
        
        for byte_tensors, icf_targets in train_loader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            optimizer1.zero_grad()
            predictions = model1(byte_tensors)
            n_pairs = len(icf_targets) // 2
            pairs = generate_ranking_pairs(icf_targets, n_pairs)
            loss = criterion1(predictions, icf_targets, pairs=pairs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
            optimizer1.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        val_loss = validate(model1, val_loader, criterion1, device)
        print(f"  Epoch {epoch+1}/5: Train={train_loss/n_batches:.4f}, Val={val_loss:.4f}")
    
    results['standard'] = {
        'model': model1,
        'final_val_loss': val_loss,
    }
    print()
    
    # Test 2: Enhanced Multi-Loss
    print("2. Training with Enhanced Multi-Loss...")
    model2 = UniversalICF().to(device)
    model2.init_weights(mean_icf=mean_icf)
    
    criterion2 = CombinedLoss  # TODO: Review - migrated from EnhancedMultiLoss()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    
    for epoch in range(5):
        model2.train()
        train_loss = 0.0
        n_batches = 0
        
        for byte_tensors, icf_targets in train_loader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            optimizer2.zero_grad()
            predictions = model2(byte_tensors)
            
            # Generate pairs and common/rare indices for multi-loss
            n_pairs = len(icf_targets) // 2
            pairs = generate_ranking_pairs(icf_targets, n_pairs)
            common_idx, rare_idx = get_common_rare_indices(icf_targets)
            
            loss = criterion2(
                predictions, icf_targets,
                pairs=pairs,
                common_indices=common_idx,
                rare_indices=rare_idx,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
            optimizer2.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        val_loss = validate(model2, val_loader, criterion1, device)  # Use standard loss for validation
        print(f"  Epoch {epoch+1}/5: Train={train_loss/n_batches:.4f}, Val={val_loss:.4f}")
    
    results['multi_loss'] = {
        'model': model2,
        'final_val_loss': val_loss,
    }
    print()
    
    # Compare
    print("=" * 70)
    print("Comparison Results")
    print("=" * 70)
    print(f"Standard Loss:  Final Val Loss = {results['standard']['final_val_loss']:.4f}")
    print(f"Multi-Loss:     Final Val Loss = {results['multi_loss']['final_val_loss']:.4f}")
    
    improvement = results['standard']['final_val_loss'] - results['multi_loss']['final_val_loss']
    if improvement > 0:
        print(f"\n✓ Multi-loss improved by {improvement:.4f} ({improvement/results['standard']['final_val_loss']*100:.1f}%)")
    elif improvement < 0:
        print(f"\n⚠ Multi-loss worse by {abs(improvement):.4f} ({abs(improvement)/results['standard']['final_val_loss']*100:.1f}%)")
    else:
        print("\n→ No significant difference")
    
    # Quick prediction comparison
    print("\nPrediction Comparison (sample words):")
    test_words = ['the', 'apple', 'xylophone', 'qzxbjk']
    for word in test_words:
        byte_seq = word.encode('utf-8')[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred1 = model1(byte_tensor).item()
            pred2 = model2(byte_tensor).item()
        
        print(f"  {word:15} Standard: {pred1:.4f}, Multi-Loss: {pred2:.4f}, Diff: {abs(pred1-pred2):.4f}")


if __name__ == "__main__":
    quick_test_multi_loss()

