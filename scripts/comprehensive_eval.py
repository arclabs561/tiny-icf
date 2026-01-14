#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
#   "scipy>=1.10.0",
# ]
# ///
"""Comprehensive evaluation with detailed error analysis."""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.eval_advanced import comprehensive_evaluation
from tiny_icf.model import UniversalICF


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Comprehensive Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Device: {device}")
    print()
    
    # Load data
    print("Loading data...")
    word_counts, total_tokens = load_frequency_list(args.data)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    test_samples = samples[:args.max_samples]
    words = [word for word, _ in test_samples]
    
    dataset = WordICFDataset(test_samples, max_length=20, augment_prob=0.0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Evaluating on {len(test_samples)} samples")
    
    # Load model
    print("Loading model...")
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    results = comprehensive_evaluation(model, dataloader, device, words=words)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    
    metrics = results["metrics"]
    print(f"\nBasic Metrics:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Spearman: {metrics['spearman_corr']:.4f}")
    print(f"  Pearson: {metrics['pearson_corr']:.4f}")
    
    print(f"\nFrequency Analysis:")
    for bin_info in results["frequency_analysis"]["bins"]:
        print(f"  {bin_info['bin']}: MAE={bin_info['mae']:.4f}, n={bin_info['n_samples']}")
    
    print(f"\nRanking Analysis:")
    rank_analysis = results["ranking_analysis"]
    print(f"  Top-100 overlap: {rank_analysis['top_k_overlap']:.1%}")
    print(f"  Bottom-100 overlap: {rank_analysis['bottom_k_overlap']:.1%}")
    print(f"  Top-100 mean rank error: {rank_analysis['top_k_mean_rank_error']:.1f}")
    print(f"  Bottom-100 mean rank error: {rank_analysis['bottom_k_mean_rank_error']:.1f}")
    
    print(f"\nWorst Predictions (Top 10):")
    for i, worst in enumerate(results["worst_predictions"][:10], 1):
        print(f"  {i}. {worst['word']:15} pred={worst['prediction']:.4f} "
              f"target={worst['target']:.4f} error={worst['error']:.4f}")
    
    # Save results
    if args.output:
        # Convert numpy arrays to lists for JSON
        json_results = {
            "metrics": results["metrics"],
            "ranking_metrics": results["ranking_metrics"],
            "frequency_analysis": results["frequency_analysis"],
            "length_analysis": results["length_analysis"],
            "ranking_analysis": results["ranking_analysis"],
            "worst_predictions": results["worst_predictions"][:50],  # Top 50
        }
        
        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

