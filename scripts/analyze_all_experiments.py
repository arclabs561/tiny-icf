#!/usr/bin/env python3
"""Analyze all running research-aligned experiments and provide insights."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

def analyze_experiment(exp_name: str) -> dict:
    """Analyze a single experiment."""
    metrics_file = PROJECT_ROOT / "models" / exp_name / "lightning_logs" / "version_0" / "metrics.csv"
    
    if not metrics_file.exists():
        return {'status': 'not_started', 'name': exp_name}
    
    try:
        df = pd.read_csv(metrics_file)
        
        # Find validation columns
        val_spearman_col = None
        for col in df.columns:
            if 'val' in col.lower() and 'spearman' in col.lower():
                val_spearman_col = col
                break
        
        if val_spearman_col and df[val_spearman_col].notna().any():
            val_df = df[df[val_spearman_col].notna()]
            latest = val_df.iloc[-1]
            
            return {
                'status': 'training',
                'name': exp_name,
                'epoch': int(latest['epoch']),
                'best_spearman': float(val_df[val_spearman_col].max()),
                'latest_spearman': float(latest[val_spearman_col]),
                'latest_mae': float(latest.get('val_mae', 0)) if 'val_mae' in latest else None,
                'latest_val_loss': float(latest.get('val_loss', 0)) if 'val_loss' in latest else None,
                'latest_train_loss': float(latest.get('train_loss_epoch', 0)) if 'train_loss_epoch' in latest else None,
                'total_rows': len(df),
            }
        else:
            # Training but no validation yet
            latest = df.iloc[-1]
            return {
                'status': 'training_no_val',
                'name': exp_name,
                'epoch': int(latest.get('epoch', 0)) if 'epoch' in latest else 0,
                'total_rows': len(df),
            }
    except Exception as e:
        return {'status': 'error', 'name': exp_name, 'error': str(e)}


def compare_experiments(results: list) -> None:
    """Compare experiments and provide insights."""
    print("📊 Research-Aligned Experiments Analysis")
    print("=" * 60)
    print()
    
    # Group by status
    training = [r for r in results if r.get('status') == 'training']
    training_no_val = [r for r in results if r.get('status') == 'training_no_val']
    not_started = [r for r in results if r.get('status') == 'not_started']
    
    if training:
        print("✅ Experiments with Validation Metrics:")
        print()
        for r in sorted(training, key=lambda x: x.get('best_spearman', 0), reverse=True):
            print(f"   {r['name']}:")
            print(f"      Epoch: {r['epoch']}")
            print(f"      Best Spearman: {r['best_spearman']:.4f}")
            print(f"      Latest Spearman: {r['latest_spearman']:.4f}")
            if r.get('latest_mae'):
                print(f"      Val MAE: {r['latest_mae']:.4f}")
            if r.get('latest_val_loss'):
                print(f"      Val Loss: {r['latest_val_loss']:.4f}")
            print()
        
        # Find best performer
        best = max(training, key=lambda x: x.get('best_spearman', 0))
        print(f"🏆 Best Performer: {best['name']} (Spearman: {best['best_spearman']:.4f})")
        print()
    
    if training_no_val:
        print("🔄 Experiments Training (no validation yet):")
        for r in training_no_val:
            print(f"   {r['name']}: Epoch {r.get('epoch', 0)}, {r.get('total_rows', 0)} metric rows")
        print()
    
    if not_started:
        print("⏳ Not Started:")
        for r in not_started:
            print(f"   {r['name']}")
        print()
    
    # Recommendations
    if training:
        avg_spearman = sum(r['best_spearman'] for r in training) / len(training)
        print(f"📈 Average Best Spearman: {avg_spearman:.4f}")
        print()
        
        if avg_spearman > 0.15:
            print("💡 Recommendations:")
            print("   - Results look promising! Continue training.")
            print("   - Consider launching iteration 2 experiments soon.")
            print("   - Monitor for overfitting (val_loss vs train_loss).")
        elif avg_spearman > 0.10:
            print("💡 Recommendations:")
            print("   - Moderate progress. Monitor closely.")
            print("   - Consider adjusting learning rate or loss weights.")
        else:
            print("💡 Recommendations:")
            print("   - Early stages. Wait for more epochs.")
            print("   - Check if learning rate is appropriate.")


def main():
    """Main entry point."""
    experiments = [
        'research_aligned_standard',
        'research_aligned_neural_sort',
        'research_aligned_high_spearman',
        'research_aligned_strong_reg',
        'research_aligned_residual',
    ]
    
    results = [analyze_experiment(exp) for exp in experiments]
    compare_experiments(results)
    
    # Save summary
    summary_path = PROJECT_ROOT / "models" / "experiment_analysis.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {summary_path}")


if __name__ == '__main__':
    main()

