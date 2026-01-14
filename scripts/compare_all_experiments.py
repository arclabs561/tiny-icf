#!/usr/bin/env python3
"""Compare all research-aligned experiments and provide detailed insights."""

import sys
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).parent.parent

def load_experiment_metrics(exp_name: str) -> dict:
    """Load and analyze experiment metrics."""
    metrics_file = PROJECT_ROOT / "models" / exp_name / "lightning_logs" / "version_0" / "metrics.csv"
    
    if not metrics_file.exists():
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        
        # Find validation columns
        val_spearman_col = None
        val_mae_col = None
        val_loss_col = None
        
        for col in df.columns:
            if 'val' in col.lower() and 'spearman' in col.lower():
                val_spearman_col = col
            if 'val' in col.lower() and 'mae' in col.lower():
                val_mae_col = col
            if 'val' in col.lower() and 'loss' in col.lower() and 'step' not in col.lower():
                val_loss_col = col
        
        result = {
            'name': exp_name,
            'total_rows': len(df),
            'has_validation': False,
        }
        
        if val_spearman_col and df[val_spearman_col].notna().any():
            val_df = df[df[val_spearman_col].notna()].copy()
            result['has_validation'] = True
            result['validation_epochs'] = len(val_df)
            result['best_spearman'] = float(val_df[val_spearman_col].max())
            result['latest_spearman'] = float(val_df[val_spearman_col].iloc[-1])
            result['latest_epoch'] = int(val_df['epoch'].iloc[-1])
            
            if val_mae_col and val_mae_col in val_df.columns:
                result['best_mae'] = float(val_df[val_mae_col].min())
                result['latest_mae'] = float(val_df[val_mae_col].iloc[-1])
            
            if val_loss_col and val_loss_col in val_df.columns:
                result['best_val_loss'] = float(val_df[val_loss_col].min())
                result['latest_val_loss'] = float(val_df[val_loss_col].iloc[-1])
            
            # Check for overfitting (val_loss increasing while train_loss decreasing)
            if 'train_loss_epoch' in val_df.columns and val_loss_col:
                train_loss = val_df['train_loss_epoch'].iloc[-1]
                val_loss = val_df[val_loss_col].iloc[-1]
                result['train_val_gap'] = float(val_loss - train_loss)
                result['potential_overfitting'] = bool(val_loss > train_loss * 1.2)
        
        return result
    except Exception as e:
        return {'name': exp_name, 'error': str(e)}


def compare_all():
    """Compare all experiments."""
    experiments = [
        'research_aligned_standard',
        'research_aligned_neural_sort',
        'research_aligned_high_spearman',
        'research_aligned_strong_reg',
        'research_aligned_residual',
    ]
    
    results = []
    for exp in experiments:
        metrics = load_experiment_metrics(exp)
        if metrics:
            results.append(metrics)
    
    # Sort by best Spearman
    results_with_val = [r for r in results if r.get('has_validation')]
    results_with_val.sort(key=lambda x: x.get('best_spearman', 0), reverse=True)
    
    print("📊 Comprehensive Experiment Comparison")
    print("=" * 70)
    print()
    
    if results_with_val:
        print("🏆 Rankings by Best Spearman Correlation:")
        print()
        for i, r in enumerate(results_with_val, 1):
            print(f"{i}. {r['name']}:")
            print(f"   Best Spearman: {r['best_spearman']:.4f}")
            print(f"   Latest Spearman: {r['latest_spearman']:.4f}")
            print(f"   Epoch: {r['latest_epoch']}")
            if 'best_mae' in r:
                print(f"   Best MAE: {r['best_mae']:.4f}")
            if 'potential_overfitting' in r and r['potential_overfitting']:
                print(f"   ⚠️  Potential overfitting detected (val_loss > train_loss * 1.2)")
            print()
        
        # Statistics
        best_spearmans = [r['best_spearman'] for r in results_with_val]
        avg_best = sum(best_spearmans) / len(best_spearmans)
        max_best = max(best_spearmans)
        min_best = min(best_spearmans)
        
        print("📈 Statistics:")
        print(f"   Average Best Spearman: {avg_best:.4f}")
        print(f"   Max Best Spearman: {max_best:.4f}")
        print(f"   Min Best Spearman: {min_best:.4f}")
        print(f"   Range: {max_best - min_best:.4f}")
        print()
        
        # Recommendations
        print("💡 Recommendations:")
        if max_best > 0.15:
            print("   ✅ Excellent progress! Best experiment is performing well.")
        elif max_best > 0.12:
            print("   ✅ Good progress. Continue training and monitoring.")
        else:
            print("   ⚠️  Early stages. Monitor closely for improvements.")
        
        if max_best - min_best > 0.02:
            print(f"   📊 Significant variation ({max_best - min_best:.4f}) suggests different configs have different potential.")
        
        # Check for overfitting
        overfitting = [r for r in results_with_val if r.get('potential_overfitting')]
        if overfitting:
            print(f"   ⚠️  {len(overfitting)} experiment(s) showing potential overfitting.")
            print("      Consider: increased regularization, early stopping, or dropout.")
        
        print()
    
    # Experiments without validation
    results_no_val = [r for r in results if not r.get('has_validation')]
    if results_no_val:
        print("🔄 Experiments Still Training (no validation yet):")
        for r in results_no_val:
            print(f"   {r['name']}: {r.get('total_rows', 0)} metric rows")
        print()
    
    # Save comparison
    comparison_path = PROJECT_ROOT / "models" / "experiment_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Comparison saved to: {comparison_path}")


if __name__ == '__main__':
    compare_all()
