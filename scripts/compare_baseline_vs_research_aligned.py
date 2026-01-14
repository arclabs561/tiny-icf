#!/usr/bin/env python3
"""Compare baseline experiments with their research-aligned counterparts."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

# Mapping of baseline -> research-aligned experiments
BASELINE_MAPPINGS = {
    'standard_improved': 'research_aligned_standard',
    'multitask_icf_high_spearman_plateau': 'research_aligned_high_spearman',
    'multitask_icf_strong_reg': 'research_aligned_strong_reg',
    'residual_listwise': 'research_aligned_residual',
}


def load_experiment_results(exp_name: str) -> Dict:
    """Load results for an experiment."""
    models_dir = PROJECT_ROOT / "models" / exp_name
    
    if not models_dir.exists():
        return {'status': 'not_started'}
    
    # Try to load from registry
    registry_path = PROJECT_ROOT / "models" / "experiment_registry.json"
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            if exp_name in registry.get('experiments', {}):
                return registry['experiments'][exp_name].get('results', {})
    
    # Fallback: check metrics CSV
    metrics_csv = models_dir / "lightning_logs" / "version_0" / "metrics.csv"
    if metrics_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(metrics_csv)
            val_df = df[df['val_spearman_corr'].notna()]
            if len(val_df) > 0:
                return {
                    'status': 'completed',
                    'best_spearman': float(val_df['val_spearman_corr'].max()),
                    'final_spearman': float(val_df['val_spearman_corr'].iloc[-1]),
                    'epochs_trained': len(val_df),
                }
        except Exception:
            pass
    
    # Check if in progress
    log_file = models_dir / "training.log"
    if log_file.exists():
        return {'status': 'in_progress'}
    
    return {'status': 'not_started'}


def compare_experiments(baseline_name: str, research_aligned_name: str) -> Dict:
    """Compare a baseline with its research-aligned counterpart."""
    baseline_results = load_experiment_results(baseline_name)
    ra_results = load_experiment_results(research_aligned_name)
    
    comparison = {
        'baseline': {
            'name': baseline_name,
            'results': baseline_results,
        },
        'research_aligned': {
            'name': research_aligned_name,
            'results': ra_results,
        },
    }
    
    # Calculate improvement if both completed
    if (baseline_results.get('status') == 'completed' and 
        ra_results.get('status') == 'completed'):
        
        baseline_spearman = baseline_results.get('best_spearman', 0)
        ra_spearman = ra_results.get('best_spearman', 0)
        
        if baseline_spearman > 0:
            improvement = ((ra_spearman - baseline_spearman) / baseline_spearman) * 100
            comparison['improvement_pct'] = improvement
            comparison['improvement_abs'] = ra_spearman - baseline_spearman
    
    return comparison


def main():
    """Main entry point."""
    print("📊 Comparing Baseline vs Research-Aligned Experiments\n")
    print("=" * 70)
    
    comparisons = []
    for baseline, ra in BASELINE_MAPPINGS.items():
        comp = compare_experiments(baseline, ra)
        comparisons.append(comp)
    
    # Print results
    for comp in comparisons:
        baseline = comp['baseline']
        ra = comp['research_aligned']
        
        print(f"\n🔬 {baseline['name']} vs {ra['name']}")
        print("-" * 70)
        
        baseline_status = baseline['results'].get('status', 'unknown')
        ra_status = ra['results'].get('status', 'unknown')
        
        print(f"Baseline:     {baseline_status}")
        if baseline_status == 'completed':
            best = baseline['results'].get('best_spearman', 0)
            final = baseline['results'].get('final_spearman', 0)
            epochs = baseline['results'].get('epochs_trained', 0)
            print(f"              Best Spearman: {best:.4f}, Final: {final:.4f}, Epochs: {epochs}")
        
        print(f"Research-RA:  {ra_status}")
        if ra_status == 'completed':
            best = ra['results'].get('best_spearman', 0)
            final = ra['results'].get('final_spearman', 0)
            epochs = ra['results'].get('epochs_trained', 0)
            print(f"              Best Spearman: {best:.4f}, Final: {final:.4f}, Epochs: {epochs}")
        
        if 'improvement_pct' in comp:
            print(f"✅ Improvement: {comp['improvement_abs']:+.4f} ({comp['improvement_pct']:+.1f}%)")
        elif baseline_status == 'completed' and ra_status != 'completed':
            print("⏳ Research-aligned experiment not yet completed")
        elif baseline_status != 'completed' and ra_status == 'completed':
            print("⚠️  Baseline not yet completed (cannot compare)")
        else:
            print("⏳ Both experiments pending")
    
    # Summary
    completed_comparisons = [c for c in comparisons if 'improvement_pct' in c]
    if completed_comparisons:
        avg_improvement = sum(c['improvement_pct'] for c in completed_comparisons) / len(completed_comparisons)
        print(f"\n📈 Average improvement: {avg_improvement:+.1f}%")
    
    # Save comparison
    output_path = PROJECT_ROOT / "models" / "baseline_vs_research_aligned.json"
    with open(output_path, 'w') as f:
        json.dump(comparisons, f, indent=2)
    
    print(f"\n💾 Comparison saved to: {output_path}")


if __name__ == '__main__':
    main()

