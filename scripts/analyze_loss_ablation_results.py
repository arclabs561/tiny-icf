#!/usr/bin/env python3
"""Analyze loss ablation experiment results to determine optimal configuration."""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def load_experiment_metrics(exp_name: str) -> Dict:
    """Load metrics for an experiment."""
    exp_dir = Path(f"models/{exp_name}/lightning_logs/version_0")
    metrics_file = exp_dir / "metrics.csv"
    
    if not metrics_file.exists():
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        val_cols = [c for c in df.columns if 'val' in c.lower() and 'spearman' in c.lower()]
        
        if not val_cols:
            return None
        
        val_df = df[df[val_cols[0]].notna()].copy()
        if len(val_df) == 0:
            return None
        
        latest = val_df.iloc[-1]
        best = val_df[val_cols[0]].max()
        epoch = int(latest['epoch']) if 'epoch' in latest else len(val_df) - 1
        
        result = {
            'name': exp_name,
            'epoch': epoch,
            'best_spearman': float(best),
            'latest_spearman': float(latest[val_cols[0]]),
        }
        
        # Add other metrics if available
        if 'val_mae' in latest and pd.notna(latest['val_mae']):
            result['val_mae'] = float(latest['val_mae'])
        if 'val_mean_squared_error' in latest and pd.notna(latest['val_mean_squared_error']):
            result['mse'] = float(latest['val_mean_squared_error'])
        if 'val_percent_close_10pct' in latest and pd.notna(latest['val_percent_close_10pct']):
            result['percent_close_10pct'] = float(latest['val_percent_close_10pct'])
        
        return result
    except Exception as e:
        print(f"⚠️  Error loading {exp_name}: {e}")
        return None


def get_experiment_config(exp_name: str) -> Dict:
    """Extract configuration from experiment name or metadata."""
    # Parse experiment name to infer configuration
    config = {'name': exp_name}
    
    if 'pure_spearman' in exp_name:
        config.update({
            'use_spearman': True,
            'spearman_weight': 10.0,
            'rank_weight': 0.0,
            'use_focal': True,
        })
    elif 'pure_ranking' in exp_name:
        config.update({
            'use_spearman': False,
            'rank_weight': 1.0,
            'use_focal': True,
        })
    elif 'balanced_hybrid' in exp_name:
        config.update({
            'use_spearman': True,
            'spearman_weight': 10.0,
            'rank_weight': 0.5,
            'use_focal': True,
        })
    elif 'high_spearman' in exp_name and 'very_high' not in exp_name:
        config.update({
            'use_spearman': True,
            'spearman_weight': 20.0,
            'rank_weight': 0.5,
            'use_focal': True,
        })
    elif 'very_high_spearman' in exp_name:
        config.update({
            'use_spearman': True,
            'spearman_weight': 50.0,
            'rank_weight': 0.1,
            'use_focal': True,
        })
    elif 'high_ranking' in exp_name:
        config.update({
            'use_spearman': True,
            'spearman_weight': 5.0,
            'rank_weight': 2.0,
            'use_focal': True,
        })
    elif 'no_focal' in exp_name:
        config.update({
            'use_spearman': True,
            'spearman_weight': 10.0,
            'rank_weight': 0.5,
            'use_focal': False,
        })
    elif 'with_monotonicity' in exp_name:
        config.update({
            'use_spearman': True,
            'spearman_weight': 10.0,
            'rank_weight': 0.5,
            'use_focal': True,
            'use_monotonicity': True,
        })
    elif 'low_spearman' in exp_name:
        config.update({
            'use_spearman': True,
            'spearman_weight': 5.0,
            'rank_weight': 1.0,
            'use_focal': True,
        })
    elif 'equal_weights' in exp_name:
        config.update({
            'use_spearman': True,
            'spearman_weight': 1.0,
            'rank_weight': 1.0,
            'use_focal': True,
        })
    
    return config


def analyze_ablation_results():
    """Analyze all loss ablation experiments."""
    
    experiments = [
        'loss_ablation_pure_spearman',
        'loss_ablation_pure_ranking',
        'loss_ablation_balanced_hybrid',
        'loss_ablation_high_spearman',
        'loss_ablation_very_high_spearman',
        'loss_ablation_high_ranking',
        'loss_ablation_no_focal',
        'loss_ablation_with_monotonicity',
        'loss_ablation_low_spearman',
        'loss_ablation_equal_weights',
    ]
    
    results = []
    for exp in experiments:
        metrics = load_experiment_metrics(exp)
        if metrics:
            config = get_experiment_config(exp)
            results.append({**metrics, **config})
    
    if not results:
        print("❌ No results found. Experiments may still be running.")
        return
    
    # Sort by best Spearman
    results.sort(key=lambda x: x['best_spearman'], reverse=True)
    
    print("=" * 80)
    print("📊 Loss Ablation Analysis")
    print("=" * 80)
    print()
    
    print("🏆 Rankings by Best Spearman Correlation:")
    print()
    print(f"{'Rank':<6} {'Experiment':<35} {'Best ρ':<10} {'Latest ρ':<10} {'Config':<30}")
    print("-" * 80)
    
    for i, r in enumerate(results, 1):
        config_str = f"Spearman={r.get('spearman_weight', 'N/A')}×, Rank={r.get('rank_weight', 'N/A')}×"
        if not r.get('use_focal', True):
            config_str += ", No Focal"
        if r.get('use_monotonicity', False):
            config_str += ", +Mono"
        
        print(f"{i:<6} {r['name']:<35} {r['best_spearman']:<10.4f} {r['latest_spearman']:<10.4f} {config_str:<30}")
    
    print()
    print("=" * 80)
    print("📈 Statistical Analysis:")
    print("=" * 80)
    print()
    
    # Group by configuration
    spearman_only = [r for r in results if r.get('rank_weight', 1) == 0.0]
    ranking_only = [r for r in results if not r.get('use_spearman', True)]
    hybrid = [r for r in results if r.get('use_spearman', True) and r.get('rank_weight', 0) > 0]
    
    if spearman_only:
        avg = np.mean([r['best_spearman'] for r in spearman_only])
        print(f"📊 Pure Spearman (no ranking): {avg:.4f} average")
    
    if ranking_only:
        avg = np.mean([r['best_spearman'] for r in ranking_only])
        print(f"📊 Pure Ranking (no Spearman): {avg:.4f} average")
    
    if hybrid:
        avg = np.mean([r['best_spearman'] for r in hybrid])
        print(f"📊 Hybrid (Spearman + Ranking): {avg:.4f} average")
    
    # Spearman weight analysis
    spearman_weights = {}
    for r in results:
        if r.get('use_spearman', False):
            weight = r.get('spearman_weight', 0)
            if weight not in spearman_weights:
                spearman_weights[weight] = []
            spearman_weights[weight].append(r['best_spearman'])
    
    if spearman_weights:
        print()
        print("📊 Spearman Weight Analysis:")
        for weight in sorted(spearman_weights.keys()):
            avg = np.mean(spearman_weights[weight])
            print(f"   Weight {weight:4.1f}×: {avg:.4f} average Spearman")
    
    # Focal loss analysis
    with_focal = [r for r in results if r.get('use_focal', True)]
    without_focal = [r for r in results if not r.get('use_focal', True)]
    
    if with_focal and without_focal:
        print()
        print("📊 Focal Loss Impact:")
        print(f"   With Focal: {np.mean([r['best_spearman'] for r in with_focal]):.4f} average")
        print(f"   Without Focal: {np.mean([r['best_spearman'] for r in without_focal]):.4f} average")
    
    # Save results
    output_path = Path('models/loss_ablation_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"💾 Analysis saved to: {output_path}")
    print()
    
    # Recommendations
    best = results[0]
    print("=" * 80)
    print("💡 Recommendations:")
    print("=" * 80)
    print()
    print(f"🏆 Best Configuration: {best['name']}")
    print(f"   Best Spearman: {best['best_spearman']:.4f}")
    print(f"   Configuration: Spearman={best.get('spearman_weight', 'N/A')}×, Rank={best.get('rank_weight', 'N/A')}×")
    print()
    
    if best['best_spearman'] > 0.20:
        print("✅ Excellent! This configuration is performing well.")
    elif best['best_spearman'] > 0.15:
        print("✅ Good progress. Consider further tuning.")
    else:
        print("⚠️  Results are below target. Consider:")
        print("   - Increasing model capacity")
        print("   - Trying different ranking methods (neural_sort, probabilistic)")
        print("   - Adding monotonicity constraints")
        print("   - Knowledge distillation")


if __name__ == '__main__':
    analyze_ablation_results()

