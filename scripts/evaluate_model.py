#!/usr/bin/env python3
"""Comprehensive model evaluation script."""

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from tiny_icf.model import UniversalICF
from tiny_icf.data import WordICFDataset, load_frequency_list, compute_normalized_icf
from tiny_icf.eval import (
    compute_metrics,
    evaluate_ranking,
    evaluate_jabberwocky,
    evaluate_on_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--data", type=Path, help="Path to frequency CSV for dataset evaluation")
    parser.add_argument("--output", type=Path, help="Path to save evaluation results JSON")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples for dataset eval")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--jabberwocky-only", action="store_true", help="Only run Jabberwocky Protocol")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("Model Evaluation: tiny-icf")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print()
    
    # Load model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print("✓ Model loaded")
    print()
    
    results = {}
    
    # 1. Jabberwocky Protocol
    print("1. Jabberwocky Protocol")
    print("-" * 80)
    jabberwocky_results = evaluate_jabberwocky(model, device)
    results['jabberwocky'] = jabberwocky_results
    
    print(f"Pass Rate: {jabberwocky_results['pass_rate']:.1%} ({jabberwocky_results['passed_count']}/{jabberwocky_results['total_count']})")
    print()
    for r in jabberwocky_results['results']:
        status = "✓" if r['passed'] else "✗"
        print(f"  {status} {r['word']:20} -> {r['predicted']:.4f} (expected: {r['min_icf']:.2f}-{r['max_icf']:.2f}) - {r['description']}")
    print()
    
    if args.jabberwocky_only:
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        return
    
    # 2. Dataset Evaluation
    if args.data and args.data.exists():
        print("2. Dataset Evaluation")
        print("-" * 80)
        print(f"Data: {args.data}")
        print(f"Max samples: {args.max_samples}")
        print()
        
        # Load dataset
        word_counts, total_tokens = load_frequency_list(args.data)
        word_icf = compute_normalized_icf(word_counts, total_tokens)
        
        # Create dataset
        pairs = list(word_icf.items())
        dataset = WordICFDataset(pairs, max_length=20)
        
        print(f"Dataset size: {len(dataset)} words")
        print("Evaluating...")
        
        # Evaluate
        eval_results = evaluate_on_dataset(
            model,
            dataset,
            device,
            max_samples=args.max_samples,
            batch_size=64,
        )
        
        results['dataset'] = eval_results
        
        # Print metrics
        metrics = eval_results['metrics']
        print()
        print("Metrics:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Median AE: {metrics['median_ae']:.4f}")
        print(f"  Max AE: {metrics['max_ae']:.4f}")
        print(f"  P95 AE: {metrics['p95_ae']:.4f}")
        print()
        print("Correlations:")
        print(f"  Spearman: {metrics['spearman_corr']:.4f} (p={metrics['spearman_p']:.4f})")
        print(f"  Pearson:  {metrics['pearson_corr']:.4f} (p={metrics['pearson_p']:.4f})")
        print(f"  Kendall:  {metrics['kendall_corr']:.4f} (p={metrics['kendall_p']:.4f})")
        print()
        print("Statistics:")
        print(f"  Predictions: mean={metrics['pred_mean']:.4f}, std={metrics['pred_std']:.4f}, range=[{metrics['pred_min']:.4f}, {metrics['pred_max']:.4f}]")
        print(f"  Targets:     mean={metrics['target_mean']:.4f}, std={metrics['target_std']:.4f}, range=[{metrics['target_min']:.4f}, {metrics['target_max']:.4f}]")
        print()
        print("Calibration:")
        print(f"  Calibration Error: {metrics['calibration_error']:.4f}")
        print()
        
        # Ranking metrics
        rank_metrics = eval_results['ranking_metrics']
        print("Ranking Quality:")
        print(f"  Precision@10: {rank_metrics['precision_at_k']:.2%}")
        print(f"  Top-10 Overlap: {rank_metrics['top_k_overlap']}/10")
        print(f"  Mean Rank Error: {rank_metrics['mean_rank_error']:.2f}")
        print()
    else:
        print("2. Dataset Evaluation: Skipped (no data file provided)")
        print()
    
    # 3. Summary
    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    
    if 'dataset' in results:
        metrics = results['dataset']['metrics']
        print(f"✓ MAE: {metrics['mae']:.4f}")
        print(f"✓ Spearman: {metrics['spearman_corr']:.4f}")
        print(f"✓ Jabberwocky: {jabberwocky_results['pass_rate']:.1%}")
        
        # Overall assessment
        print()
        print("Assessment:")
        if metrics['mae'] < 0.1:
            print("  ✓ Excellent MAE (< 0.1)")
        elif metrics['mae'] < 0.25:
            print("  ✓ Good MAE (< 0.25)")
        else:
            print("  ⚠ MAE could be improved")
        
        if metrics['spearman_corr'] > 0.8:
            print("  ✓ Excellent correlation (> 0.8)")
        elif metrics['spearman_corr'] > 0.6:
            print("  ✓ Good correlation (> 0.6)")
        else:
            print("  ⚠ Correlation could be improved")
    
    if jabberwocky_results['pass_rate'] >= 0.8:
        print("  ✓ Excellent Jabberwocky performance (≥ 80%)")
    elif jabberwocky_results['pass_rate'] >= 0.6:
        print("  ✓ Good Jabberwocky performance (≥ 60%)")
    else:
        print("  ⚠ Jabberwocky performance could be improved")
    
    print()
    
    # Save results
    if args.output:
        # Convert numpy arrays to lists for JSON
        if 'dataset' in results:
            results['dataset']['predictions'] = results['dataset']['predictions'].tolist()
            results['dataset']['targets'] = results['dataset']['targets'].tolist()
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()

