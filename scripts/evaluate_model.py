#!/usr/bin/env -S uv run
"""Comprehensive model evaluation script."""

import argparse
import json
import random
from pathlib import Path

import torch
import numpy as np

from tiny_icf.calibration import load_calibration
from tiny_icf.checkpoint import load_model
from tiny_icf.data import (
    WordICFDataset,
    load_frequency_list,
    compute_normalized_icf,
    stratified_sample,
)
from tiny_icf.eval import (
    compute_metrics,
    evaluate_ranking,
    evaluate_jabberwocky,
    evaluate_on_dataset,
)
from tiny_icf.oov_calibration import DEFAULT_SATURATION_FIX, SaturationFixConfig


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--data", type=Path, help="Path to frequency CSV for dataset evaluation")
    parser.add_argument("--output", type=Path, help="Path to save evaluation results JSON")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples for dataset eval")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--jabberwocky-only", action="store_true", help="Only run Jabberwocky Protocol")
    parser.add_argument(
        "--saturation-fix",
        action="store_true",
        help="Optional OOV-focused fix: unsaturate clamp-to-1.0 outputs using raw_output.",
    )
    parser.add_argument(
        "--fix-center",
        type=float,
        default=float(DEFAULT_SATURATION_FIX.center),
        help="Saturation-fix center parameter (raw_output at which fixed score is ~0.5).",
    )
    parser.add_argument(
        "--fix-scale",
        type=float,
        default=float(DEFAULT_SATURATION_FIX.scale),
        help="Saturation-fix scale parameter (smaller = steeper mapping).",
    )
    parser.add_argument(
        "--fix-conf-weight",
        type=float,
        default=float(DEFAULT_SATURATION_FIX.confidence_weight),
        help="Optional saturation-fix confidence weight (0 disables).",
    )
    parser.add_argument(
        "--icf-mode",
        type=str,
        default="log",
        choices=["log", "rank"],
        help="Target definition for dataset evaluation: 'log' or 'rank'.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Path to calibration JSON (a, b). Apply learned affine calibration to predictions.",
    )

    args = parser.parse_args()
    random.seed(42)
    
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("Model Evaluation: tiny-icf")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print()
    
    # Load model
    model, _checkpoint = load_model(args.model, device=device)
    model.eval()
    print("✓ Model loaded")

    calibration = load_calibration(args.calibration) if args.calibration else None
    if args.calibration and calibration is None:
        raise SystemExit(f"Calibration file not found or invalid: {args.calibration}")
    if calibration:
        print(f"✓ Calibration loaded from {args.calibration}")
    print()

    results = {}

    # 1. Jabberwocky Protocol
    print("1. Jabberwocky Protocol")
    print("-" * 80)
    fix_config = SaturationFixConfig(
        eps=float(DEFAULT_SATURATION_FIX.eps),
        center=float(args.fix_center),
        scale=float(args.fix_scale),
        confidence_weight=float(args.fix_conf_weight),
        confidence_center=float(DEFAULT_SATURATION_FIX.confidence_center),
    )
    jabberwocky_results = evaluate_jabberwocky(
        model,
        device,
        saturation_fix=bool(args.saturation_fix),
        saturation_fix_config=fix_config,
        calibration=calibration,
    )
    results['jabberwocky'] = jabberwocky_results
    
    print(f"Pass Rate: {jabberwocky_results['pass_rate']:.1%} ({jabberwocky_results['passed_count']}/{jabberwocky_results['total_count']})")
    print()
    print(f"  {'word':22s}  {'pred':>6}  {'expected':>14}  ok?  description")
    print(f"  {'-'*22}  {'-'*6}  {'-'*14}  ---  -----------")
    for r in jabberwocky_results['results']:
        status = "✓" if r['passed'] else "✗"
        exp = f"[{r['min_icf']:.2f}, {r['max_icf']:.2f}]"
        print(f"  {status} {r['word']:22s}  {r['predicted']:6.4f}  {exp:>14}       {r['description']}")
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
        word_icf = compute_normalized_icf(word_counts, total_tokens, mode=args.icf_mode)
        
        # Create dataset
        # Important: `word_icf` iteration order follows the input file (usually sorted by count),
        # so a naive `pairs[:max_samples]` would evaluate only the head (very narrow target range).
        # Use the same stratified sampling strategy as training to get a representative slice.
        pairs = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
        random.shuffle(pairs)
        dataset = WordICFDataset(pairs, max_length=20)
        
        print(f"Dataset size: {len(dataset)} words")
        print("Evaluating...")
        
        # Evaluate (optionally with learned calibration)
        eval_results = evaluate_on_dataset(
            model,
            dataset,
            device,
            max_samples=args.max_samples,
            batch_size=64,
            calibration=calibration,
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

