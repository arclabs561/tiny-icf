#!/usr/bin/env -S uv run
"""Compare different training approaches (standard vs multi-loss)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.eval import compute_metrics, evaluate_jabberwocky
from tiny_icf.model import UniversalICF


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """Evaluate model and return metrics."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for byte_tensors, icf_targets in dataloader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            predictions = model(byte_tensors)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(icf_targets.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return compute_metrics(predictions, targets)


def compare_models(
    model1_path: Path,
    model2_path: Path,
    data_path: Path,
    model1_name: str = "Standard",
    model2_name: str = "Multi-Loss",
):
    """Compare two trained models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    # Use validation split
    split_idx = int(len(samples) * 0.8)
    val_samples = samples[split_idx:]
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Load models
    model1 = UniversalICF().to(device)
    model1.load_state_dict(torch.load(model1_path, map_location=device))
    
    model2 = UniversalICF().to(device)
    model2.load_state_dict(torch.load(model2_path, map_location=device))
    
    # Evaluate
    print(f"\n{'='*70}")
    print(f"Comparing {model1_name} vs {model2_name}")
    print(f"{'='*70}\n")
    
    print(f"Evaluating {model1_name}...")
    metrics1 = evaluate_model(model1, val_loader, device)
    jabberwocky1 = evaluate_jabberwocky(model1, device)
    
    print(f"\nEvaluating {model2_name}...")
    metrics2 = evaluate_model(model2, val_loader, device)
    jabberwocky2 = evaluate_jabberwocky(model2, device)
    
    # Print comparison
    print(f"\n{'='*70}")
    print("Metrics Comparison")
    print(f"{'='*70}\n")
    print(f"{'Metric':<25} {model1_name:<20} {model2_name:<20} {'Winner':<10}")
    print("-" * 70)
    
    comparisons = [
        ("MAE", metrics1["mae"], metrics2["mae"], "lower"),
        ("RMSE", metrics1["rmse"], metrics2["rmse"], "lower"),
        ("Spearman", metrics1["spearman"], metrics2["spearman"], "higher"),
        ("Pearson", metrics1["pearson"], metrics2["pearson"], "higher"),
        ("Kendall", metrics1["kendall"], metrics2["kendall"], "higher"),
    ]
    
    for name, val1, val2, better in comparisons:
        if better == "lower":
            winner = model1_name if val1 < val2 else model2_name
            diff = abs(val1 - val2)
        else:
            winner = model1_name if val1 > val2 else model2_name
            diff = abs(val1 - val2)
        
        print(f"{name:<25} {val1:<20.6f} {val2:<20.6f} {winner:<10} (Δ={diff:.6f})")
    
    # Jabberwocky comparison
    print(f"\n{'='*70}")
    print("Jabberwocky Protocol")
    print(f"{'='*70}\n")
    print(f"{'Test':<25} {model1_name:<20} {model2_name:<20}")
    print("-" * 70)
    
    for word in jabberwocky1["results"]:
        score1 = jabberwocky1["results"][word]["score"]
        score2 = jabberwocky2["results"][word]["score"]
        passed1 = jabberwocky1["results"][word]["passed"]
        passed2 = jabberwocky2["results"][word]["passed"]
        
        status1 = "✓" if passed1 else "✗"
        status2 = "✓" if passed2 else "✗"
        
        print(f"{word:<25} {status1} {score1:<18.4f} {status2} {score2:<18.4f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}\n")
    
    jabberwocky_passed1 = sum(1 for r in jabberwocky1["results"].values() if r["passed"])
    jabberwocky_passed2 = sum(1 for r in jabberwocky2["results"].values() if r["passed"])
    
    print(f"{model1_name}:")
    print(f"  - MAE: {metrics1['mae']:.6f}")
    print(f"  - Spearman: {metrics1['spearman']:.6f}")
    print(f"  - Jabberwocky: {jabberwocky_passed1}/{len(jabberwocky1['results'])} passed")
    
    print(f"\n{model2_name}:")
    print(f"  - MAE: {metrics2['mae']:.6f}")
    print(f"  - Spearman: {metrics2['spearman']:.6f}")
    print(f"  - Jabberwocky: {jabberwocky_passed2}/{len(jabberwocky2['results'])} passed")


def main():
    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument("--model1", type=Path, required=True, help="Path to first model")
    parser.add_argument("--model2", type=Path, required=True, help="Path to second model")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument("--name1", type=str, default="Standard", help="Name for first model")
    parser.add_argument("--name2", type=str, default="Multi-Loss", help="Name for second model")
    
    args = parser.parse_args()
    
    compare_models(
        args.model1,
        args.model2,
        args.data,
        args.name1,
        args.name2,
    )


if __name__ == "__main__":
    main()

