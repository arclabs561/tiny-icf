#!/usr/bin/env -S uv run
"""Compare multiple trained models side-by-side."""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.eval import compute_metrics, evaluate_jabberwocky, evaluate_ranking
from tiny_icf.eval_advanced import comprehensive_evaluation
from tiny_icf.model import UniversalICF


def load_model(model_path: Path, device: torch.device):
    """Load a trained model."""
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def compare_models(models: dict, dataloader: DataLoader, device: torch.device, words: list):
    """Compare multiple models on the same dataset."""
    results = {}
    
    for name, model_path in models.items():
        print(f"\nEvaluating {name}...")
        model = load_model(model_path, device)
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for byte_tensors, icf_targets in dataloader:
                byte_tensors = byte_tensors.to(device)
                icf_targets = icf_targets.to(device)
                predictions = model(byte_tensors)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(icf_targets.cpu().numpy())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        metrics = compute_metrics(preds, targets)
        ranking = evaluate_ranking(preds, targets, top_k=10)
        jabberwocky = evaluate_jabberwocky(model, device)
        
        results[name] = {
            "metrics": metrics,
            "ranking": ranking,
            "jabberwocky": jabberwocky,
            "predictions": preds,
            "targets": targets,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare multiple trained models")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths (format: name:path)")
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
    
    # Parse models
    models = {}
    for model_spec in args.models:
        if ":" in model_spec:
            name, path = model_spec.split(":", 1)
            models[name] = Path(path)
        else:
            # Use filename as name
            path = Path(model_spec)
            models[path.stem] = path
    
    print("=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(f"Comparing {len(models)} models:")
    for name, path in models.items():
        print(f"  {name}: {path}")
    print()
    
    # Load data
    word_counts, total_tokens = load_frequency_list(args.data)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    test_samples = samples[:args.max_samples]
    words = [word for word, _ in test_samples]
    
    dataset = WordICFDataset(test_samples, max_length=20, augment_prob=0.0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Compare models
    results = compare_models(models, dataloader, device, words)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    
    print(f"\n{'Model':<20} {'MAE':<8} {'RMSE':<8} {'Spearman':<10} {'Jabberwocky':<15}")
    print("-" * 70)
    
    best_spearman = -1
    best_model = None
    
    for name, result in results.items():
        metrics = result["metrics"]
        jabberwocky = result["jabberwocky"]
        jabberwocky_str = f"{jabberwocky['passed_count']}/{jabberwocky['total_count']}"
        
        print(f"{name:<20} {metrics['mae']:<8.4f} {metrics['rmse']:<8.4f} "
              f"{metrics['spearman_corr']:<10.4f} {jabberwocky_str:<15}")
        
        if metrics['spearman_corr'] > best_spearman:
            best_spearman = metrics['spearman_corr']
            best_model = name
    
    print(f"\n✓ Best model: {best_model} (Spearman: {best_spearman:.4f})")
    
    # Detailed comparison
    print(f"\n{'='*70}")
    print("Detailed Metrics")
    print(f"{'='*70}\n")
    
    for name, result in results.items():
        metrics = result["metrics"]
        ranking = result["ranking"]
        jabberwocky = result["jabberwocky"]
        preds = result["predictions"]
        
        print(f"{name}:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Spearman: {metrics['spearman_corr']:.4f}")
        print(f"  Pearson: {metrics['pearson_corr']:.4f}")
        print(f"  Precision@10: {ranking['precision_at_k']:.4f}")
        print(f"  Jabberwocky: {jabberwocky['passed_count']}/{jabberwocky['total_count']} ({jabberwocky['pass_rate']:.1%})")
        print(f"  Predictions: mean={preds.mean():.4f}, std={preds.std():.4f}, "
              f"range=[{preds.min():.4f}, {preds.max():.4f}]")
        print()
    
    # Save results
    if args.output:
        json_results = {}
        for name, result in results.items():
            json_results[name] = {
                "metrics": result["metrics"],
                "ranking": result["ranking"],
                "jabberwocky": result["jabberwocky"],
                "prediction_stats": {
                    "mean": float(result["predictions"].mean()),
                    "std": float(result["predictions"].std()),
                    "min": float(result["predictions"].min()),
                    "max": float(result["predictions"].max()),
                },
            }
        
        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

