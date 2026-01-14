#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas>=2.0.0",
#   "aim>=3.0.0",
# ]
# ///
"""
Compare experiments using Aim and trainctl utilities.

Provides systematic comparison of training experiments:
- Best metrics across experiments
- Hyperparameter effects
- Training progress
- Resource usage
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

# Add trainctl utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'trainctl' / 'utils'))

try:
    from aim import Run
    HAS_AIM = True
except ImportError:
    HAS_AIM = False
    print("⚠️  Aim not available. Install with: uv pip install aim")

try:
    from trainctl.utils.metrics_loader import (
        compare_experiments as trainctl_compare,
        list_experiments,
        get_experiment_status,
    )
    from trainctl.utils.storage_manager import get_storage_usage
    HAS_TRAINCTL_UTILS = True
except ImportError:
    HAS_TRAINCTL_UTILS = False
    print("⚠️  trainctl utilities not available.")


def compare_via_aim(experiment_name: str = "icf-training", limit: int = 50) -> List[Dict]:
    """
    Compare experiments using Aim.
    
    Args:
        experiment_name: Aim experiment name
        limit: Maximum number of runs to compare
    
    Returns:
        List of run dictionaries with metrics
    """
    if not HAS_AIM:
        return []
    
    runs = Run.filter(experiment=experiment_name, limit=limit)
    
    results = []
    for run in runs:
        # Get hyperparameters
        hparams = run.get('hparams', {}) if hasattr(run, 'get') else {}
        
        # Get best metrics
        metrics = {}
        for metric_name in ['val_spearman_corr', 'val_mae', 'val_rmse', 'val_loss']:
            try:
                metric_series = run.metrics(metric_name)
                if metric_series:
                    # Get best value (highest for spearman, lowest for others)
                    values = [m.value for m in metric_series]
                    if values:
                        if metric_name == 'val_spearman_corr':
                            metrics[metric_name] = max(values)
                        else:
                            metrics[metric_name] = min(values)
            except Exception:
                pass
        
        results.append({
            'run_name': run.name if hasattr(run, 'name') else str(run.hash),
            'run_hash': run.hash if hasattr(run, 'hash') else None,
            'hparams': hparams,
            'metrics': metrics,
        })
    
    # Sort by best spearman correlation
    results.sort(key=lambda x: x['metrics'].get('val_spearman_corr', -1.0), reverse=True)
    
    return results


def compare_via_trainctl(experiment_names: Optional[List[str]] = None) -> List[Dict]:
    """
    Compare experiments using trainctl utilities.
    
    Args:
        experiment_names: Optional list of experiment names. If None, compares all.
    
    Returns:
        List of experiment status dictionaries
    """
    if not HAS_TRAINCTL_UTILS:
        return []
    
    if experiment_names is None:
        experiment_names = list_experiments()
    
    return trainctl_compare(experiment_names)


def print_comparison_table(results: List[Dict], source: str = "aim"):
    """Print a formatted comparison table."""
    if not results:
        print(f"No results from {source}")
        return
    
    print(f"\n{'='*80}")
    print(f"Experiment Comparison ({source.upper()})")
    print(f"{'='*80}\n")
    
    # Header
    if source == "aim":
        print(f"{'Run Name':<30} {'Best Spearman':>15} {'Best MAE':>12} {'Best RMSE':>12}")
        print("-" * 80)
        for r in results[:20]:  # Top 20
            metrics = r.get('metrics', {})
            print(f"{r['run_name'][:28]:<30} "
                  f"{metrics.get('val_spearman_corr', 0.0):>15.4f} "
                  f"{metrics.get('val_mae', 0.0):>12.4f} "
                  f"{metrics.get('val_rmse', 0.0):>12.4f}")
    else:  # trainctl
        print(f"{'Experiment':<30} {'Status':<12} {'Best Spearman':>15} {'Epochs':>8}")
        print("-" * 80)
        for r in results[:20]:  # Top 20
            print(f"{r.get('name', 'unknown')[:28]:<30} "
                  f"{r.get('status', 'unknown')[:10]:<12} "
                  f"{r.get('best_spearman', 0.0):>15.4f} "
                  f"{r.get('epochs_trained', 0):>8}")


def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare training experiments")
    parser.add_argument('--method', choices=['aim', 'trainctl', 'both'], default='both',
                       help='Comparison method')
    parser.add_argument('--experiment', default='icf-training',
                       help='Aim experiment name')
    parser.add_argument('--experiments', nargs='+',
                       help='Specific experiment names to compare (for trainctl)')
    parser.add_argument('--limit', type=int, default=50,
                       help='Maximum number of runs to compare (Aim)')
    parser.add_argument('--storage', action='store_true',
                       help='Show storage usage statistics')
    
    args = parser.parse_args()
    
    # Compare via Aim
    if args.method in ['aim', 'both']:
        if HAS_AIM:
            aim_results = compare_via_aim(args.experiment, args.limit)
            print_comparison_table(aim_results, source='aim')
        else:
            print("⚠️  Aim not available. Skipping Aim comparison.")
    
    # Compare via trainctl
    if args.method in ['trainctl', 'both']:
        if HAS_TRAINCTL_UTILS:
            trainctl_results = compare_via_trainctl(args.experiments)
            print_comparison_table(trainctl_results, source='trainctl')
        else:
            print("⚠️  trainctl utilities not available. Skipping trainctl comparison.")
    
    # Storage usage
    if args.storage and HAS_TRAINCTL_UTILS:
        print(f"\n{'='*80}")
        print("Storage Usage Statistics")
        print(f"{'='*80}\n")
        stats = get_storage_usage()
        print(f"Total experiments: {stats['num_experiments']}")
        print(f"Total checkpoints: {stats['num_checkpoints']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")
        if stats['experiments']:
            print("\nTop 10 largest experiments:")
            for i, exp in enumerate(stats['experiments'][:10], 1):
                print(f"{i:2d}. {exp['name']:<40} {exp['size_mb']:>8.2f} MB ({exp['num_checkpoints']} checkpoints)")


if __name__ == '__main__':
    main()
