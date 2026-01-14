#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
#   "tqdm>=4.65.0",
#   "scipy>=1.10.0",
#   "aim>=3.29.0",
# ]
# ///
"""
Ablation study: Compare different loss configurations.

Tests:
1. Huber only (rank_weight=0)
2. Current (rank_weight=2.0, pairwise)
3. High ranking weight (rank_weight=10.0, pairwise)
4. Listwise (LambdaRank)
5. Listwise (ApproxNDCG)
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.loss import CombinedLoss
from tiny_icf.loss import CombinedLoss  # TODO: Review - migrated from CombinedListwiseLoss
from tiny_icf.loss import CombinedLoss  # TODO: Review - migrated from ResearchBasedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.eval import compute_metrics
from tiny_icf.eval_rbo import compute_rbo_metrics

try:
    from tiny_icf.aim_tracker import AimTracker
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    AimTracker = None


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_config(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 20,
    config_name: str = "config",
) -> dict:
    """Train a single configuration and return results."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    best_spearman = -1.0
    best_metrics = None
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for byte_tensors, icf_targets in train_loader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(byte_tensors)
            
            # Collapse detection
            if predictions.std().item() < 0.01:
                print(f"  ⚠ Collapse detected at epoch {epoch}")
                break
            
            # Compute loss
            if isinstance(criterion, CombinedLoss):
                from tiny_icf.train import generate_ranking_pairs
                n_pairs = min(len(icf_targets), 32)
                pairs, pair_diffs = generate_ranking_pairs(
                    icf_targets, n_pairs, min_diff=0.05, use_weighted_sampling=True
                )
                loss = criterion(
                    predictions, icf_targets,
                    pairs=pairs,
                    pair_target_diffs=pair_diffs,
                    smooth_ranking=True,
                )
            elif isinstance(criterion, CombinedLoss  # TODO: Review - migrated from CombinedListwiseLoss):
                # Listwise losses don't need pairs
                loss = criterion(predictions, icf_targets)
            else:
                # CombinedLoss  # TODO: Review - migrated from ResearchBasedLoss and others
                loss = criterion(predictions, icf_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validate
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for byte_tensors, icf_targets in val_loader:
                    byte_tensors = byte_tensors.to(device)
                    icf_targets = icf_targets.to(device)
                    predictions = model(byte_tensors)
                    all_preds.append(predictions.cpu())
                    all_targets.append(icf_targets.cpu())
            
            predictions = torch.cat(all_preds).numpy()
            targets = torch.cat(all_targets).numpy()
            
            metrics = compute_metrics(predictions, targets)
            
            # Add RBO
            try:
                rbo_metrics = compute_rbo_metrics(
                    torch.tensor(predictions),
                    torch.tensor(targets),
                )
                metrics.update(rbo_metrics)
            except Exception:
                pass
            
            spearman = metrics.get('spearman_corr', 0.0)
            if spearman > best_spearman:
                best_spearman = spearman
                best_metrics = metrics.copy()
                best_metrics['epoch'] = epoch
    
    return best_metrics or {}


def main():
    parser = argparse.ArgumentParser(description="Ablation study: loss configurations")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per configuration")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", type=Path, default=Path("ablation_results.json"), help="Output JSON")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--aim", action="store_true", help="Enable Aim experiment tracking")
    parser.add_argument("--aim-experiment", type=str, default="ablation-study", help="Aim experiment name")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*70)
    print("Loss Configuration Ablation Study")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    word_counts, total_tokens = load_frequency_list(args.data)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    train_dataset = WordICFDataset(train_samples, max_length=20, augment_prob=0.1)
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    mean_icf = np.mean([icf for _, icf in train_samples])
    
    # Configurations to test
    from tiny_icf.loss import CombinedLoss  # TODO: Review - migrated from DifferentiableSortingLoss, check_differentiable_sorting_available
    
    available_sorting = check_differentiable_sorting_available()
    
    configs = [
        {
            "name": "huber_only",
            "criterion": CombinedLoss(huber_delta=0.1, rank_weight=0.0),
            "description": "Huber loss only (no ranking)",
        },
        {
            "name": "pairwise_rank_2.0",
            "criterion": CombinedLoss(huber_delta=0.1, rank_weight=2.0, rank_margin=0.1),
            "description": "Pairwise ranking (weight=2.0, current)",
        },
        {
            "name": "pairwise_rank_10.0",
            "criterion": CombinedLoss(huber_delta=0.1, rank_weight=10.0, rank_margin=0.1),
            "description": "Pairwise ranking (weight=10.0, high)",
        },
        {
            "name": "listwise_lambdarank",
            "criterion": CombinedLoss  # TODO: Review - migrated from CombinedListwiseLoss(
                huber_delta=0.1,
                listwise_weight=1.0,
                listwise_method="lambdarank",
            ),
            "description": "Listwise LambdaRank",
        },
        {
            "name": "listwise_approx_ndcg",
            "criterion": CombinedLoss  # TODO: Review - migrated from CombinedListwiseLoss(
                huber_delta=0.1,
                listwise_weight=1.0,
                listwise_method="approx_ndcg",
            ),
            "description": "Listwise ApproxNDCG",
        },
        {
            "name": "research_neural_ndcg",
            "criterion": CombinedLoss  # TODO: Review - migrated from ResearchBasedLoss(
                use_neural_ndcg=True,
                use_softmax_ce=False,
                use_focal=False,
                ndcg_weight=0.5,
            ),
            "description": "Research: NeuralNDCG (direct NDCG optimization)",
        },
        {
            "name": "research_softmax_ce",
            "criterion": CombinedLoss  # TODO: Review - migrated from ResearchBasedLoss(
                use_neural_ndcg=False,
                use_softmax_ce=True,
                use_focal=False,
                softmax_weight=0.3,
            ),
            "description": "Research: Softmax Cross-Entropy Ranking",
        },
        {
            "name": "research_combined",
            "criterion": CombinedLoss  # TODO: Review - migrated from ResearchBasedLoss(
                use_neural_ndcg=True,
                use_softmax_ce=True,
                use_focal=False,
                ndcg_weight=0.5,
                softmax_weight=0.3,
            ),
            "description": "Research: NeuralNDCG + Softmax CE combined",
        },
        {
            "name": "research_full",
            "criterion": CombinedLoss  # TODO: Review - migrated from ResearchBasedLoss(
                use_neural_ndcg=True,
                use_softmax_ce=True,
                use_focal=True,
                ndcg_weight=0.5,
                softmax_weight=0.3,
                focal_weight=0.2,
            ),
            "description": "Research: Full combination (NDCG + Softmax CE + Focal)",
        },
    ]
    
    # Add differentiable sorting if available
    if available_sorting.get("diffsort"):
        configs.append({
            "name": "diffsort",
            "criterion": CombinedLoss  # TODO: Review - migrated from DifferentiableSortingLoss(
                method="diffsort",
                steepness=5.0,
                huber_weight=0.3,
            ),
            "description": "Differentiable sorting (diffsort) - direct Spearman optimization",
        })
    
    if available_sorting.get("fast_soft_sort"):
        configs.append({
            "name": "fast_soft_sort",
            "criterion": CombinedLoss  # TODO: Review - migrated from DifferentiableSortingLoss(
                method="fast_soft_sort",
                regularization_strength=1.0,
                huber_weight=0.3,
            ),
            "description": "Differentiable sorting (fast-soft-sort) - direct Spearman optimization",
        })
    
    results = {}
    
    # Initialize Aim tracker if requested
    aim_tracker = None
    if args.aim and AIM_AVAILABLE:
        aim_tracker = AimTracker(
            experiment_name=args.aim_experiment,
            run_name="ablation-study",
        )
        aim_tracker.track_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_configs": len(configs),
            "device": str(device),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
        })
        print(f"\n✓ Aim tracking enabled (experiment: {args.aim_experiment})")
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"Configuration {i}/{len(configs)}: {config['name']}")
        print(f"Description: {config['description']}")
        print("="*70)
        
        # Create fresh model
        set_seed(42)  # Same seed for fair comparison
        model = UniversalICF().to(device)
        model.init_weights(mean_icf=mean_icf)
        
        # Train
        metrics = train_config(
            model,
            train_loader,
            val_loader,
            config['criterion'],
            device,
            epochs=args.epochs,
            config_name=config['name'],
        )
        
        results[config['name']] = {
            "description": config['description'],
            "metrics": metrics,
        }
        
        print(f"\nResults for {config['name']}:")
        print(f"  Spearman: {metrics.get('spearman_corr', 0.0):.4f}")
        print(f"  MAE: {metrics.get('mae', 0.0):.4f}")
        print(f"  RBO (full): {metrics.get('rbo_full', 0.0):.4f}")
        print(f"  Pred std: {metrics.get('pred_std', 0.0):.4f}")
        
        # Track with Aim
        if aim_tracker:
            aim_tracker.track_metric(
                f"{config['name']}_spearman",
                metrics.get('spearman_corr', 0.0),
                step=i
            )
            aim_tracker.track_metric(
                f"{config['name']}_mae",
                metrics.get('mae', 0.0),
                step=i
            )
            aim_tracker.track_metric(
                f"{config['name']}_rbo",
                metrics.get('rbo_full', 0.0),
                step=i
            )
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"{name:25s} | Spearman: {metrics.get('spearman_corr', 0.0):.4f} | "
              f"RBO: {metrics.get('rbo_full', 0.0):.4f} | "
              f"MAE: {metrics.get('mae', 0.0):.4f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    if aim_tracker:
        aim_tracker.close()
        print(f"\n✓ Aim tracking complete. View with: aim up")


if __name__ == "__main__":
    main()

