#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
# ]
# ///

"""
Comprehensive analysis of the latest training experiments.
Analyzes metrics, training curves, and provides insights.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_metrics_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load Lightning metrics CSV."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        # Filter to validation metrics only (where val_spearman_corr is not NaN)
        df = df[df['val_spearman_corr'].notna()].copy()
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def analyze_experiment(exp_name: str, summary: Dict, metrics_df: Optional[pd.DataFrame]) -> Dict:
    """Analyze a single experiment."""
    analysis = {
        'name': exp_name,
        'summary': summary,
        'metrics': None,
        'convergence': None,
        'training_stability': None,
    }
    
    if metrics_df is not None and len(metrics_df) > 0:
        # Training curve analysis
        spearman_vals = metrics_df['val_spearman_corr'].values
        mae_vals = metrics_df['val_mae'].values
        loss_vals = metrics_df['val_loss'].values
        
        analysis['metrics'] = {
            'max_spearman': float(spearman_vals.max()),
            'final_spearman': float(spearman_vals[-1]),
            'best_epoch': int(spearman_vals.argmax()),
            'min_mae': float(mae_vals.min()),
            'final_mae': float(mae_vals[-1]),
            'min_loss': float(loss_vals.min()),
            'final_loss': float(loss_vals[-1]),
            'total_epochs': len(spearman_vals),
        }
        
        # Convergence analysis
        if len(spearman_vals) > 10:
            # Check if model plateaued
            last_10 = spearman_vals[-10:].values if hasattr(spearman_vals, 'values') else spearman_vals[-10:]
            improvement_last_10 = float(np.max(last_10) - np.min(last_10))
            first_10 = spearman_vals[:10].values if hasattr(spearman_vals, 'values') else spearman_vals[:10]
            improvement_total = float(spearman_vals.max() - np.mean(first_10))
            
            analysis['convergence'] = {
                'plateaued': improvement_last_10 < 0.01,
                'improvement_last_10': float(improvement_last_10),
                'improvement_total': float(improvement_total),
                'convergence_rate': float(improvement_total / len(spearman_vals)) if len(spearman_vals) > 0 else 0.0,
            }
        
        # Training stability (variance in metrics)
        if len(spearman_vals) > 5:
            analysis['training_stability'] = {
                'spearman_std': float(spearman_vals.std()),
                'mae_std': float(mae_vals.std()),
                'loss_std': float(loss_vals.std()),
                'spearman_range': float(spearman_vals.max() - spearman_vals.min()),
            }
    
    return analysis


def main():
    summary_path = Path("models/experiment_summary.json")
    if not summary_path.exists():
        print("❌ No experiment summary found. Run training first.")
        return
    
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    print("=" * 80)
    print("COMPREHENSIVE EXPERIMENT ANALYSIS")
    print("=" * 80)
    
    # Load metrics for each experiment
    analyses = []
    for exp_summary in summary_data:
        exp_name = exp_summary['name']
        metrics_path = Path("models") / exp_name / "lightning_logs" / "version_0" / "metrics.csv"
        metrics_df = load_metrics_csv(metrics_path)
        analysis = analyze_experiment(exp_name, exp_summary, metrics_df)
        analyses.append(analysis)
    
    # 1. Overall Summary
    print("\n📊 OVERALL SUMMARY")
    print("-" * 80)
    print(f"{'Experiment':<25} {'Best Spearman':<15} {'Final Spearman':<15} {'Final MAE':<12} {'Epochs':<8}")
    print("-" * 80)
    
    for a in analyses:
        name = a['name']
        summary = a['summary']
        metrics = a.get('metrics', {})
        best = metrics.get('max_spearman', summary.get('final_spearman', 0))
        final = summary.get('final_spearman', 0)
        mae = summary.get('final_mae', 0)
        epochs = summary.get('epochs_trained', 0)
        print(f"{name:<25} {best:<15.4f} {final:<15.4f} {mae:<12.4f} {epochs:<8}")
    
    # 2. Best Experiment
    best_exp = max(analyses, key=lambda x: x.get('metrics', {}).get('max_spearman', x['summary'].get('final_spearman', -1)))
    print(f"\n🏆 BEST EXPERIMENT: {best_exp['name']}")
    if best_exp.get('metrics'):
        m = best_exp['metrics']
        print(f"   Best Spearman: {m['max_spearman']:.4f} (epoch {m['best_epoch']})")
        print(f"   Final Spearman: {m['final_spearman']:.4f}")
        print(f"   Final MAE: {m['final_mae']:.4f}")
        print(f"   Total Epochs: {m['total_epochs']}")
    
    # 3. Training Curve Analysis
    print("\n📈 TRAINING CURVE ANALYSIS")
    print("-" * 80)
    
    for a in analyses:
        if not a.get('metrics'):
            continue
        
        name = a['name']
        m = a['metrics']
        conv = a.get('convergence', {})
        stability = a.get('training_stability', {})
        
        print(f"\n{name}:")
        print(f"  Peak Performance: Spearman={m['max_spearman']:.4f} at epoch {m['best_epoch']}")
        print(f"  Final Performance: Spearman={m['final_spearman']:.4f}, MAE={m['final_mae']:.4f}")
        
        if conv:
            print(f"  Convergence: {'Plateaued' if conv.get('plateaued') else 'Still improving'}")
            print(f"    Total improvement: {conv.get('improvement_total', 0):.4f}")
            print(f"    Last 10 epochs improvement: {conv.get('improvement_last_10', 0):.4f}")
        
        if stability:
            print(f"  Stability: Spearman std={stability.get('spearman_std', 0):.4f}, range={stability.get('spearman_range', 0):.4f}")
    
    # 4. Configuration Comparison
    print("\n⚙️  CONFIGURATION COMPARISON")
    print("-" * 80)
    
    # Load configs from training script
    configs = {
        'spearman_aggressive': {
            'spearman_weight': 50.0,
            'spearman_reg_strength': 2.0,
            'rank_weight': 0.001,
            'lr': 1e-3,
            'batch_size': 256,
            'model': 'universal',
        },
        'spearman_sharp': {
            'spearman_weight': 20.0,
            'spearman_reg_strength': 5.0,
            'rank_weight': 0.01,
            'lr': 8e-4,
            'batch_size': 256,
            'model': 'universal',
        },
        'residual_spearman': {
            'spearman_weight': 15.0,
            'spearman_reg_strength': 1.5,
            'rank_weight': 0.01,
            'lr': 1e-3,
            'batch_size': 256,
            'model': 'residual',
        },
    }
    
    print(f"{'Config':<25} {'Spearman W':<12} {'Reg Strength':<12} {'LR':<10} {'Model':<10} {'Result':<12}")
    print("-" * 80)
    
    for a in analyses:
        name = a['name']
        config = configs.get(name, {})
        result = a.get('metrics', {}).get('max_spearman', a['summary'].get('final_spearman', 0))
        print(f"{name:<25} "
              f"{config.get('spearman_weight', 0):<12.1f} "
              f"{config.get('spearman_reg_strength', 0):<12.1f} "
              f"{config.get('lr', 0):<10.4f} "
              f"{config.get('model', 'unknown'):<10} "
              f"{result:<12.4f}")
    
    # 5. Key Insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 80)
    
    # Find patterns
    universal_exps = [a for a in analyses if 'residual' not in a['name']]
    residual_exps = [a for a in analyses if 'residual' in a['name']]
    
    if universal_exps:
        avg_universal = np.mean([a.get('metrics', {}).get('max_spearman', a['summary'].get('final_spearman', 0)) 
                                 for a in universal_exps])
        print(f"Universal model average: {avg_universal:.4f}")
    
    if residual_exps:
        avg_residual = np.mean([a.get('metrics', {}).get('max_spearman', a['summary'].get('final_spearman', 0)) 
                               for a in residual_exps])
        print(f"Residual model average: {avg_residual:.4f}")
        if universal_exps:
            diff = avg_residual - avg_universal
            print(f"Residual advantage: {diff:+.4f}")
    
    # Weight analysis
    high_weight = [a for a in analyses if configs.get(a['name'], {}).get('spearman_weight', 0) >= 20]
    low_weight = [a for a in analyses if configs.get(a['name', {}]).get('spearman_weight', 0) < 20]
    
    if high_weight and low_weight:
        avg_high = np.mean([a.get('metrics', {}).get('max_spearman', a['summary'].get('final_spearman', 0)) 
                           for a in high_weight])
        avg_low = np.mean([a.get('metrics', {}).get('max_spearman', a['summary'].get('final_spearman', 0)) 
                          for a in low_weight])
        print(f"\nHigh Spearman weight (≥20) average: {avg_high:.4f}")
        print(f"Low Spearman weight (<20) average: {avg_low:.4f}")
        print(f"Difference: {avg_high - avg_low:+.4f}")
    
    # 6. Recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("-" * 80)
    
    best_result = best_exp.get('metrics', {}).get('max_spearman', best_exp['summary'].get('final_spearman', 0))
    
    if best_result < 0.2:
        print("⚠️  All experiments show low Spearman correlation (<0.2)")
        print("   Consider:")
        print("   - Increasing model capacity (more layers/width)")
        print("   - Different loss formulations")
        print("   - Data augmentation or more training data")
        print("   - Different architectures")
    
    if residual_exps and universal_exps:
        if avg_residual > avg_universal:
            print("✅ Residual model shows better performance")
            print("   Consider exploring more residual configurations")
    
    # Check for overfitting
    for a in analyses:
        if a.get('convergence', {}).get('plateaued'):
            print(f"⚠️  {a['name']} plateaued early - may need different hyperparameters")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

