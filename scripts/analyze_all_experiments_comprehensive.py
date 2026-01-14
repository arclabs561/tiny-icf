#!/usr/bin/env python3
"""Comprehensive analysis of all experiments (loss ablation, iter3, distillation)."""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
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
        for metric in ['val_mae', 'val_mean_squared_error', 'val_percent_close_10pct']:
            if metric in latest and pd.notna(latest[metric]):
                result[metric.replace('val_', '')] = float(latest[metric])
        
        return result
    except Exception as e:
        print(f"⚠️  Error loading {exp_name}: {e}")
        return None


def analyze_all_experiments():
    """Analyze all experiment types."""
    print("=" * 70)
    print("📊 Comprehensive Experiment Analysis")
    print("=" * 70)
    print()
    
    # Find all experiment directories
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ models/ directory not found")
        return
    
    all_experiments = []
    for exp_dir in models_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "lightning_logs" / "version_0" / "metrics.csv").exists():
            metrics = load_experiment_metrics(exp_dir.name)
            if metrics:
                all_experiments.append(metrics)
    
    if not all_experiments:
        print("⚠️  No completed experiments found")
        return
    
    # Group by experiment type
    loss_ablation = [e for e in all_experiments if e['name'].startswith('loss_ablation_')]
    iter3 = [e for e in all_experiments if e['name'].startswith('iter3_')]
    distillation = [e for e in all_experiments if 'distillation' in e['name']]
    research_aligned = [e for e in all_experiments if e['name'].startswith('research_aligned_')]
    other = [e for e in all_experiments if not any([
        e['name'].startswith('loss_ablation_'),
        e['name'].startswith('iter3_'),
        'distillation' in e['name'],
        e['name'].startswith('research_aligned_'),
    ])]
    
    # Sort all by best Spearman
    all_experiments.sort(key=lambda x: x.get('best_spearman', 0), reverse=True)
    
    print("🏆 Top 10 Experiments (by Best Spearman):")
    print("-" * 70)
    print(f"{'Rank':<6} {'Name':<40} {'Best ρ':<10} {'Latest ρ':<10} {'Epoch':<8}")
    print("-" * 70)
    for i, exp in enumerate(all_experiments[:10], 1):
        print(f"{i:<6} {exp['name']:<40} {exp['best_spearman']:<10.4f} {exp['latest_spearman']:<10.4f} {exp['epoch']:<8}")
    print()
    
    # Summary by type
    print("📊 Summary by Experiment Type:")
    print("-" * 70)
    for exp_type, exps in [
        ("Loss Ablation", loss_ablation),
        ("Iteration 3", iter3),
        ("Distillation", distillation),
        ("Research Aligned", research_aligned),
        ("Other", other),
    ]:
        if exps:
            best = max(exps, key=lambda x: x.get('best_spearman', 0))
            avg = np.mean([e.get('best_spearman', 0) for e in exps])
            print(f"{exp_type:<20} {len(exps):<4} experiments | Best: {best['name']:<35} ({best['best_spearman']:.4f}) | Avg: {avg:.4f}")
    print()
    
    # Save comprehensive analysis
    analysis = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_experiments': len(all_experiments),
        'top_10': all_experiments[:10],
        'by_type': {
            'loss_ablation': loss_ablation,
            'iter3': iter3,
            'distillation': distillation,
            'research_aligned': research_aligned,
            'other': other,
        },
        'best_overall': all_experiments[0] if all_experiments else None,
    }
    
    output_file = Path("models/comprehensive_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"💾 Comprehensive analysis saved to: {output_file}")
    print()
    print("💡 Next Steps:")
    if iter3:
        best_iter3 = max(iter3, key=lambda x: x.get('best_spearman', 0))
        print(f"   - Best Iter3: {best_iter3['name']} ({best_iter3['best_spearman']:.4f})")
    if all_experiments:
        print(f"   - Best Overall: {all_experiments[0]['name']} ({all_experiments[0]['best_spearman']:.4f})")
        print(f"   - Consider fine-tuning around: {all_experiments[0]['name']}")


if __name__ == "__main__":
    analyze_all_experiments()
