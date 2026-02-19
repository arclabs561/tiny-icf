# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
#   "scipy>=1.10.0",
# ]
# ///
"""
Comprehensive evaluation script with baseline comparisons.

Evaluates model against:
- Character unigram baseline
- Character bigram baseline
- Word length baseline
- TFIDF baseline (if available)
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.checkpoint import load_model
from tiny_icf.data import WordICFDataset, load_frequency_list, compute_normalized_icf
from tiny_icf.eval import evaluate_on_dataset
from tiny_icf.baselines import evaluate_baselines


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with baseline comparisons")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument(
        "--icf-mode",
        type=str,
        default="log",
        choices=["log", "rank"],
        help="Target definition: 'log' (corpus ICF) or 'rank' (corpus-invariant quantile)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results.json"), help="Output JSON path"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {args.data}...")
    word_counts, total_tokens = load_frequency_list(args.data)
    icf_scores = compute_normalized_icf(word_counts, total_tokens, mode=args.icf_mode)

    # Create dataset
    samples = [(word, icf) for word, icf in icf_scores.items()]
    if args.max_samples:
        samples = samples[: args.max_samples]

    dataset = WordICFDataset(samples, max_length=20, augment_prob=0.0)

    # Load model
    print(f"Loading model from {args.model}...")
    model, _checkpoint = load_model(args.model, device=device)
    model.eval()

    # Evaluate model
    print("Evaluating model...")
    model_results = evaluate_on_dataset(model, dataset, device, args.max_samples, args.batch_size)

    # Get words and predictions
    words = model_results["words"]
    targets = model_results["targets"]

    # Evaluate baselines
    print("Evaluating baselines...")
    true_icf_dict = {word: float(target) for word, target in zip(words, targets)}
    baseline_results = evaluate_baselines(words, true_icf_dict, word_counts, total_tokens)

    # Compile results
    results = {
        "model": {
            "metrics": model_results["metrics"],
            "ranking_metrics": model_results.get("ranking_metrics", {}),
            "stratified": model_results.get("stratified", {}),
        },
        "baselines": baseline_results,
        "comparison": {
            "model_spearman": model_results["metrics"].get("spearman_corr", 0.0),
            "best_baseline": max(baseline_results.items(), key=lambda x: x[1].get("spearman", 0.0)),
        },
    }

    # Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print("\nModel Performance:")
    print(f"  Spearman: {model_results['metrics'].get('spearman_corr', 0.0):.4f}")
    print(f"  MAE: {model_results['metrics'].get('mae', 0.0):.4f}")
    print(f"  RMSE: {model_results['metrics'].get('rmse', 0.0):.4f}")
    if "ece" in model_results["metrics"]:
        print(f"  ECE: {model_results['metrics']['ece']:.4f}")

    print("\nBaseline Performance:")
    for name, metrics in baseline_results.items():
        print(
            f"  {name}: Spearman={metrics.get('spearman', 0.0):.4f}, MAE={metrics.get('mae', 0.0):.4f}"
        )

    best_baseline_name, best_baseline_metrics = results["comparison"]["best_baseline"]
    model_spearman = results["comparison"]["model_spearman"]
    best_baseline_spearman = best_baseline_metrics.get("spearman", 0.0)

    print("\nComparison:")
    print(f"  Model vs Best Baseline ({best_baseline_name}):")
    print(f"    Model: {model_spearman:.4f}")
    print(f"    Baseline: {best_baseline_spearman:.4f}")
    print(f"    Improvement: {model_spearman - best_baseline_spearman:+.4f}")

    if model_spearman > best_baseline_spearman:
        print("  ✓ Model outperforms best baseline!")
    else:
        print("  ✗ Model does not outperform best baseline")
        print("    This may indicate dataset size limitations (research suggests")
        print("    character-level CNNs need 100k-500k+ examples to outperform baselines)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
