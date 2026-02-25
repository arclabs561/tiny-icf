#!/usr/bin/env -S uv run
"""Analyze training dynamics: loss components, gradients, and learning patterns."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.eval import compute_metrics
from tiny_icf.loss import CombinedLoss, huber_loss, ranking_loss
from tiny_icf.model import UniversalICF
from tiny_icf.train import generate_ranking_pairs, set_seed


def analyze_batch(
    model: nn.Module,
    byte_tensors: torch.Tensor,
    icf_targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Analyze a single batch in detail."""
    byte_tensors = byte_tensors.to(device)
    icf_targets = icf_targets.to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(byte_tensors)
    
    # Compute individual loss components
    huber = huber_loss(predictions, icf_targets, delta=0.1)
    
    # Ranking loss
    n_pairs = min(len(icf_targets), 32)
    pairs, pair_diffs = generate_ranking_pairs(
        icf_targets, n_pairs, min_diff=0.05, use_weighted_sampling=True
    )
    rank = torch.tensor(0.0, device=device)
    if len(pairs) > 0:
        idx1, idx2 = pairs[:, 0], pairs[:, 1]
        rank = ranking_loss(
            predictions[idx1],
            predictions[idx2],
            margin=0.1,
            target_diff=pair_diffs,
            smooth=True,
        )
    
    # Total loss
    total_loss = criterion(
        predictions, icf_targets,
        pairs=pairs,
        pair_target_diffs=pair_diffs,
        smooth_ranking=True,
    )
    
    # Gradient analysis
    model.train()
    predictions = model(byte_tensors)
    loss = criterion(
        predictions, icf_targets,
        pairs=pairs,
        pair_target_diffs=pair_diffs,
        smooth_ranking=True,
    )
    loss.backward()
    
    # Compute gradient statistics
    grad_norms = []
    grad_maxes = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_max = param.grad.abs().max().item()
            grad_norms.append(grad_norm)
            grad_maxes.append(grad_max)
    
    # Clear gradients
    model.zero_grad()
    
    return {
        "huber_loss": float(huber.item()),
        "ranking_loss": float(rank.item()),
        "total_loss": float(total_loss.item()),
        "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "max_grad_norm": float(np.max(grad_norms)) if grad_norms else 0.0,
        "mean_grad_max": float(np.mean(grad_maxes)) if grad_maxes else 0.0,
        "prediction_mean": float(predictions.mean().item()),
        "prediction_std": float(predictions.std().item()),
        "prediction_min": float(predictions.min().item()),
        "prediction_max": float(predictions.max().item()),
        "target_mean": float(icf_targets.mean().item()),
        "target_std": float(icf_targets.std().item()),
        "n_pairs": len(pairs),
        "mean_pair_diff": float(pair_diffs.mean().item()) if len(pair_diffs) > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze training dynamics")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches to analyze")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    
    args = parser.parse_args()
    
    set_seed(42)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Analyzing training dynamics on {device}")
    
    # Load data
    word_counts, total_tokens = load_frequency_list(args.data)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    test_samples = samples[:5000]
    dataset = WordICFDataset(test_samples, max_length=20, augment_prob=0.0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Load model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")
    
    criterion = CombinedLoss()
    
    # Analyze batches
    print(f"\nAnalyzing {args.n_batches} batches...")
    batch_analyses = []
    
    for i, (byte_tensors, icf_targets) in enumerate(tqdm(dataloader, total=args.n_batches)):
        if i >= args.n_batches:
            break
        
        analysis = analyze_batch(model, byte_tensors, icf_targets, criterion, device)
        analysis["batch"] = i + 1
        batch_analyses.append(analysis)
    
    # Aggregate statistics
    print("\n" + "=" * 70)
    print("Training Dynamics Analysis")
    print("=" * 70)
    
    huber_losses = [a["huber_loss"] for a in batch_analyses]
    ranking_losses = [a["ranking_loss"] for a in batch_analyses]
    total_losses = [a["total_loss"] for a in batch_analyses]
    grad_norms = [a["mean_grad_norm"] for a in batch_analyses]
    pred_stds = [a["prediction_std"] for a in batch_analyses]
    
    print(f"\nLoss Components:")
    print(f"  Huber Loss:     mean={np.mean(huber_losses):.4f}, std={np.std(huber_losses):.4f}")
    print(f"  Ranking Loss:   mean={np.mean(ranking_losses):.4f}, std={np.std(ranking_losses):.4f}")
    print(f"  Total Loss:     mean={np.mean(total_losses):.4f}, std={np.std(total_losses):.4f}")
    
    print(f"\nGradients:")
    print(f"  Mean Grad Norm: mean={np.mean(grad_norms):.4f}, std={np.std(grad_norms):.4f}")
    print(f"  Max Grad Norm:  {max(a['max_grad_norm'] for a in batch_analyses):.4f}")
    
    print(f"\nPredictions:")
    print(f"  Mean Std:       {np.mean(pred_stds):.4f}")
    print(f"  Mean Range:     [{min(a['prediction_min'] for a in batch_analyses):.4f}, "
          f"{max(a['prediction_max'] for a in batch_analyses):.4f}]")
    
    print(f"\nRanking Pairs:")
    mean_pairs = np.mean([a["n_pairs"] for a in batch_analyses])
    mean_pair_diff = np.mean([a["mean_pair_diff"] for a in batch_analyses])
    print(f"  Mean Pairs/Batch: {mean_pairs:.1f}")
    print(f"  Mean Pair Diff:   {mean_pair_diff:.4f}")
    
    # Save results
    if args.output:
        results = {
            "batch_analyses": batch_analyses,
            "aggregate_stats": {
                "huber_loss": {"mean": float(np.mean(huber_losses)), "std": float(np.std(huber_losses))},
                "ranking_loss": {"mean": float(np.mean(ranking_losses)), "std": float(np.std(ranking_losses))},
                "total_loss": {"mean": float(np.mean(total_losses)), "std": float(np.std(total_losses))},
                "grad_norm": {"mean": float(np.mean(grad_norms)), "std": float(np.std(grad_norms))},
                "prediction_std": {"mean": float(np.mean(pred_stds)), "std": float(np.std(pred_stds))},
            },
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

