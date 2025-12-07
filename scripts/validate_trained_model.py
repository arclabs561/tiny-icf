#!/usr/bin/env -S uv run
"""
Validate trained model with comprehensive metrics and Jabberwocky protocol.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not available, correlation metrics will be limited")

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list
from tiny_icf.model import UniversalICF


def jabberwocky_test(model: nn.Module, device: torch.device):
    """Test model on Jabberwocky protocol words."""
    model.eval()
    
    test_cases = [
        ("the", 0.0, 0.1, "Common stopword"),
        ("xylophone", 0.7, 0.95, "Rare but valid word"),
        ("flimjam", 0.6, 0.85, "Rare, looks English"),
        ("qzxbjk", 0.95, 1.0, "Impossible structure"),
        ("unfriendliness", 0.4, 0.7, "Composed of common parts"),
    ]
    
    results = []
    print("\n🧪 Jabberwocky Protocol Test:")
    print("=" * 60)
    
    for word, min_icf, max_icf, description in test_cases:
        byte_seq = word.encode("utf-8")[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            icf = model(byte_tensor).item()
        
        passed = min_icf <= icf <= max_icf
        status = "✓" if passed else "✗"
        results.append((word, icf, passed, description))
        print(f"  {status} {word:20} -> {icf:.4f} (expected: {min_icf:.2f}-{max_icf:.2f}) - {description}")
    
    passed_count = sum(1 for _, _, p, _ in results if p)
    print(f"\n  Result: {passed_count}/{len(results)} tests passed")
    return passed_count, len(results)


def validate_on_dataset(model: nn.Module, dataset: WordICFDataset, device: torch.device, max_samples=1000):
    """Validate model on dataset and compute metrics."""
    model.eval()
    
    predictions = []
    targets = []
    words = []
    
    sample_count = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), sample_count, replace=False)
    
    with torch.no_grad():
        for idx in indices:
            byte_tensor, icf_target = dataset[idx]
            byte_tensor = byte_tensor.unsqueeze(0).to(device)
            icf_target = torch.tensor(icf_target).to(device)
            
            pred = model(byte_tensor).item()
            predictions.append(pred)
            targets.append(icf_target.item())
            
            # Get word if possible
            word_bytes = byte_tensor.cpu().squeeze().numpy()
            word = bytes(word_bytes[word_bytes > 0]).decode('utf-8', errors='ignore')
            words.append(word)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'pred_min': float(predictions.min()),
        'pred_max': float(predictions.max()),
        'pred_mean': float(predictions.mean()),
        'pred_std': float(predictions.std()),
        'target_min': float(targets.min()),
        'target_max': float(targets.max()),
        'target_mean': float(targets.mean()),
        'target_std': float(targets.std()),
    }
    
    if HAS_SCIPY and len(predictions) > 1:
        corr, p_value = spearmanr(predictions, targets)
        metrics['spearman_corr'] = corr
        metrics['spearman_p'] = p_value
    else:
        metrics['spearman_corr'] = 0.0
        metrics['spearman_p'] = 1.0
    
    return metrics, words, predictions, targets


def main():
    parser = argparse.ArgumentParser(description="Validate trained model")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--data", type=Path, help="Path to validation data CSV")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples for validation")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔍 Validating Model: {args.model}")
    print(f"   Device: {device}")
    print("=" * 60)
    
    # Load model
    model = UniversalICF().to(device)
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"✅ Model loaded: {model.count_parameters():,} parameters")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Jabberwocky test
    jabberwocky_passed, jabberwocky_total = jabberwocky_test(model, device)
    
    # Dataset validation
    if args.data and args.data.exists():
        print("\n📊 Dataset Validation:")
        print("=" * 60)
        
        word_counts, total_tokens = load_frequency_list(args.data)
        word_icf = compute_normalized_icf(word_counts, total_tokens)
        
        # Create dataset
        samples = list(word_icf.items())[:args.max_samples]
        dataset = WordICFDataset(samples, max_length=20, augment_prob=0.0)
        
        metrics, words, predictions, targets = validate_on_dataset(model, dataset, device, args.max_samples)
        
        print(f"\n  Samples tested: {len(predictions)}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        if metrics.get('spearman_corr'):
            print(f"  Spearman Correlation: {metrics['spearman_corr']:.4f} (p={metrics['spearman_p']:.4f})")
            if metrics['spearman_corr'] > 0.8:
                print("  ⭐ Excellent correlation!")
            elif metrics['spearman_corr'] > 0.6:
                print("  ✓ Good correlation")
            else:
                print("  ⚠️  Low correlation - model may need more training")
        
        print(f"\n  Prediction range: [{metrics['pred_min']:.4f}, {metrics['pred_max']:.4f}]")
        print(f"  Prediction std: {metrics['pred_std']:.4f}")
        print(f"  Target range: [{metrics['target_min']:.4f}, {metrics['target_max']:.4f}]")
        print(f"  Target std: {metrics['target_std']:.4f}")
        
        # Check for near-constant predictions
        if metrics['pred_std'] < 0.1:
            print("\n  ⚠️  WARNING: Model predictions are near-constant!")
            print("     This suggests the model is not learning properly.")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Validation Summary:")
    print(f"   Jabberwocky: {jabberwocky_passed}/{jabberwocky_total} passed")
    if args.data:
        print(f"   Spearman Correlation: {metrics.get('spearman_corr', 0):.4f}")
        if metrics.get('spearman_corr', 0) > 0.8 and jabberwocky_passed == jabberwocky_total:
            print("   ✅ Model validation PASSED")
        else:
            print("   ⚠️  Model validation needs improvement")


if __name__ == "__main__":
    main()
