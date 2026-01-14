#!/usr/bin/env -S uv run
"""Run mid-training evaluation on current model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.eval import compute_metrics, evaluate_jabberwocky
from tiny_icf.model import UniversalICF


def evaluate_model_mid_training(
    model_path: Path,
    data_path: Path,
    max_samples: int = 1000,
    device: str = "auto",
) -> dict:
    """Evaluate model mid-training."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    # Use validation split
    split_idx = int(len(samples) * 0.8)
    val_samples = samples[split_idx:split_idx + max_samples]
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    # Evaluate
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for byte_tensors, icf_targets in torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False):
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            predictions = model(byte_tensors)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(icf_targets.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    jabberwocky = evaluate_jabberwocky(model, device)
    
    # Prediction distribution stats
    pred_stats = {
        'mean': float(predictions.mean()),
        'std': float(predictions.std()),
        'min': float(predictions.min()),
        'max': float(predictions.max()),
        'range': float(predictions.max() - predictions.min()),
    }
    
    target_stats = {
        'mean': float(targets.mean()),
        'std': float(targets.std()),
        'min': float(targets.min()),
        'max': float(targets.max()),
        'range': float(targets.max() - targets.min()),
    }
    
    return {
        'metrics': metrics,
        'jabberwocky': jabberwocky,
        'prediction_stats': pred_stats,
        'target_stats': target_stats,
        'range_ratio': pred_stats['range'] / target_stats['range'] if target_stats['range'] > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Mid-training evaluation")
    parser.add_argument("--model", type=Path, required=True, help="Path to model")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples to evaluate")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Mid-Training Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print()
    
    results = evaluate_model_mid_training(args.model, args.data, args.max_samples, args.device)
    
    # Print results
    metrics = results['metrics']
    jabberwocky = results['jabberwocky']
    pred_stats = results['prediction_stats']
    target_stats = results['target_stats']
    
    print("Metrics:")
    print(f"  MAE:  {metrics['mae']:.4f} (target: <0.1)")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Spearman: {metrics['spearman_corr']:.4f} (target: >0.8)")
    print(f"  Pearson:  {metrics['pearson_corr']:.4f} (target: >0.8)")
    print()
    
    print("Prediction Distribution:")
    print(f"  Mean: {pred_stats['mean']:.4f} (target mean: {target_stats['mean']:.4f})")
    print(f"  Std:  {pred_stats['std']:.4f} (target std: {target_stats['std']:.4f})")
    print(f"  Range: [{pred_stats['min']:.4f}, {pred_stats['max']:.4f}] (target: [0.0, 1.0])")
    print(f"  Range ratio: {results['range_ratio']:.2%} (target: >80%)")
    print()
    
    print("Jabberwocky Protocol:")
    print(f"  Pass rate: {jabberwocky['passed_count']}/{jabberwocky['total_count']} ({jabberwocky['pass_rate']:.1%})")
    for r in jabberwocky['results']:
        status = "✓" if r['passed'] else "✗"
        print(f"  {status} {r['word']:15} -> {r['predicted']:.4f} (expected: {r['min_icf']:.2f}-{r['max_icf']:.2f})")
    print()
    
    # Assessment
    print("Assessment:")
    if pred_stats['std'] < 0.03:
        print("  ⚠ Prediction range too compressed (std < 0.03)")
    elif pred_stats['std'] > 0.05:
        print("  ✓ Prediction range expanding (std > 0.05)")
    else:
        print("  → Prediction range improving (std 0.03-0.05)")
    
    if metrics['spearman_corr'] < 0.3:
        print("  ⚠ Correlation very low (< 0.3)")
    elif metrics['spearman_corr'] > 0.6:
        print("  ✓ Correlation good (> 0.6)")
    else:
        print("  → Correlation improving (0.3-0.6)")
    
    if results['range_ratio'] < 0.5:
        print("  ⚠ Prediction range too narrow (< 50% of target range)")
    elif results['range_ratio'] > 0.8:
        print("  ✓ Prediction range good (> 80% of target range)")
    else:
        print("  → Prediction range improving (50-80%)")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()

