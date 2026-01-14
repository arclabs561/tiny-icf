#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas>=2.0.0",
# ]
# ///
"""
Analyze loss component bounds from experiment logs.

This script reads metrics.csv files and compares loss components
against theoretical bounds to identify optimization issues.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import argparse

# Add trainctl utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'trainctl' / 'utils'))

try:
    from trainctl.utils.metrics_loader import (
        list_experiments,
        get_experiment_dir,
        load_metrics_csv,
        get_validation_metrics,
    )
    HAS_TRAINCTL_UTILS = True
except ImportError:
    HAS_TRAINCTL_UTILS = False
    print("❌ trainctl utilities not available.")
    sys.exit(1)


# Theoretical bounds from LOSS_COMPONENT_BOUNDS.md (refined thresholds)
COMPONENT_BOUNDS = {
    'huber': {'good': 0.08, 'poor': 0.20, 'best': 0.05},  # Refined: top performers achieve 0.05-0.08
    'rank': {'good': 0.12, 'poor': 0.30, 'best': 0.05},  # Refined: top performers achieve 0.08-0.12
    'spearman': {'good': 0.82, 'poor': 0.90, 'best': 0.81},  # Refined: top performers achieve 0.81-0.82
    'asymmetric_penalty': {'good': 0.05, 'poor': 0.10, 'best': 0.02},
    'monotonicity': {'good': 0.01, 'poor': 0.05, 'best': 0.0},
    'quantile': {'good': 0.20, 'poor': 0.30, 'best': 0.10},
}


def analyze_component_bounds(
    experiment_names: Optional[List[str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Analyzes loss component bounds for specified experiments.

    Args:
        experiment_names: List of experiment names to analyze. If None, all experiments.
        verbose: If True, print detailed analysis for each experiment.

    Returns:
        DataFrame with bounds analysis results
    """
    if not HAS_TRAINCTL_UTILS:
        return pd.DataFrame()

    if experiment_names is None:
        experiment_names = list_experiments()

    if not experiment_names:
        print("No experiments found to analyze.")
        return pd.DataFrame()

    results = []
    
    for exp_name in experiment_names:
        exp_dir = get_experiment_dir(exp_name)
        df = load_metrics_csv(exp_dir)

        if df is None or df.empty:
            if verbose:
                print(f"⚠️  {exp_name}: No metrics.csv")
            continue

        val_df = get_validation_metrics(df)
        if val_df.empty:
            if verbose:
                print(f"⚠️  {exp_name}: No validation metrics")
            continue

        # Get best Spearman correlation
        best_spearman = val_df['val_spearman_corr'].max() if 'val_spearman_corr' in val_df.columns else None

        # Find loss component columns
        loss_cols = [col for col in val_df.columns if col.startswith('val_loss_') and 
                    not col.endswith('_status') and not col.endswith('_vs_good') and 
                    col != 'val_loss' and col != 'val_loss_step' and col != 'val_loss_epoch']
        
        if not loss_cols:
            if verbose:
                print(f"⚠️  {exp_name}: No loss components logged")
            continue

        # Get latest values
        latest_row = val_df.iloc[-1]
        
        component_analysis = {}
        for col in loss_cols:
            component_name = col.replace('val_loss_', '')
            component_value = latest_row.get(col)
            
            if pd.isna(component_value):
                continue
            
            # Compare to bounds
            if component_name in COMPONENT_BOUNDS:
                bounds = COMPONENT_BOUNDS[component_name]
                value = float(component_value)
                
                if value <= bounds['best']:
                    status = 'best'
                elif value <= bounds['good']:
                    status = 'good'
                elif value <= bounds['poor']:
                    status = 'acceptable'
                else:
                    status = 'poor'
                
                component_analysis[component_name] = {
                    'value': value,
                    'status': status,
                    'vs_good': value / bounds['good'] if bounds['good'] > 0 else None,
                    'vs_best': value / bounds['best'] if bounds['best'] > 0 else None,
                }
            else:
                # Unknown component, just record value
                component_analysis[component_name] = {
                    'value': float(component_value),
                    'status': 'unknown',
                    'vs_good': None,
                    'vs_best': None,
                }
        
        if component_analysis:
            # Create summary
            status_counts = {}
            for comp_data in component_analysis.values():
                status = comp_data['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            results.append({
                'experiment': exp_name,
                'best_spearman': best_spearman,
                'components_analyzed': len(component_analysis),
                'best_count': status_counts.get('best', 0),
                'good_count': status_counts.get('good', 0),
                'acceptable_count': status_counts.get('acceptable', 0),
                'poor_count': status_counts.get('poor', 0),
                'unknown_count': status_counts.get('unknown', 0),
                'components': component_analysis,
            })
    
    if not results:
        return pd.DataFrame()
    
    # Create summary DataFrame
    summary_data = []
    for r in results:
        summary_data.append({
            'experiment': r['experiment'],
            'best_spearman': r['best_spearman'],
            'components': r['components_analyzed'],
            'best': r['best_count'],
            'good': r['good_count'],
            'acceptable': r['acceptable_count'],
            'poor': r['poor_count'],
            'unknown': r['unknown_count'],
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if verbose:
        print("\n📊 Detailed Component Analysis\n")
        print("=" * 80)
        for r in results:
            print(f"\n🔍 {r['experiment']}")
            print(f"   Best Spearman: {r['best_spearman']:.4f if r['best_spearman'] else 'N/A'}")
            print(f"   Components: {r['components_analyzed']}")
            print(f"   Status: {r['best_count']} best, {r['good_count']} good, "
                  f"{r['acceptable_count']} acceptable, {r['poor_count']} poor")
            
            for comp_name, comp_data in r['components'].items():
                status_emoji = {
                    'best': '✅',
                    'good': '✓',
                    'acceptable': '⚠️',
                    'poor': '❌',
                    'unknown': '?'
                }.get(comp_data['status'], '?')
                
                print(f"   {status_emoji} {comp_name}: {comp_data['value']:.4f} ({comp_data['status']})")
                if comp_data['vs_good'] is not None:
                    print(f"      vs good: {comp_data['vs_good']:.2f}x")
    
    return summary_df


def find_optimization_issues(experiment_names: Optional[List[str]] = None) -> List[Dict]:
    """
    Find experiments with optimization issues based on bounds.

    Args:
        experiment_names: List of experiment names to check. If None, all experiments.

    Returns:
        List of issues found
    """
    if not HAS_TRAINCTL_UTILS:
        return []

    if experiment_names is None:
        experiment_names = list_experiments()

    issues = []
    
    for exp_name in experiment_names:
        exp_dir = get_experiment_dir(exp_name)
        df = load_metrics_csv(exp_dir)

        if df is None or df.empty:
            continue

        val_df = get_validation_metrics(df)
        if val_df.empty:
            continue

        latest_row = val_df.iloc[-1]
        
        # Check each component
        for component_name, bounds in COMPONENT_BOUNDS.items():
            col = f'val_loss_{component_name}'
            if col not in val_df.columns:
                continue
            
            value = latest_row.get(col)
            if pd.isna(value):
                continue
            
            value = float(value)
            
            # Check for issues
            if value > bounds['poor']:
                issues.append({
                    'experiment': exp_name,
                    'component': component_name,
                    'value': value,
                    'threshold': bounds['poor'],
                    'severity': 'poor',
                    'message': f"{component_name} loss ({value:.4f}) exceeds poor threshold ({bounds['poor']:.4f})"
                })
            elif value > bounds['good']:
                issues.append({
                    'experiment': exp_name,
                    'component': component_name,
                    'value': value,
                    'threshold': bounds['good'],
                    'severity': 'acceptable',
                    'message': f"{component_name} loss ({value:.4f}) exceeds good threshold ({bounds['good']:.4f})"
                })
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="Analyze loss component bounds from experiment logs")
    parser.add_argument('--experiments', nargs='+',
                       help='Specific experiments to analyze (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed analysis for each experiment')
    parser.add_argument('--issues', action='store_true',
                       help='Find and report optimization issues')
    
    args = parser.parse_args()
    
    if args.issues:
        issues = find_optimization_issues(experiment_names=args.experiments)
        
        if not issues:
            print("✅ No optimization issues found!")
            return
        
        print(f"\n⚠️  Found {len(issues)} optimization issues:\n")
        print("=" * 80)
        
        # Group by severity
        poor_issues = [i for i in issues if i['severity'] == 'poor']
        acceptable_issues = [i for i in issues if i['severity'] == 'acceptable']
        
        if poor_issues:
            print("\n❌ Poor Performance Issues:")
            for issue in poor_issues:
                print(f"   {issue['experiment']}: {issue['message']}")
        
        if acceptable_issues:
            print("\n⚠️  Acceptable Performance Issues:")
            for issue in acceptable_issues:
                print(f"   {issue['experiment']}: {issue['message']}")
    else:
        summary_df = analyze_component_bounds(
            experiment_names=args.experiments,
            verbose=args.verbose
        )
        
        if summary_df.empty:
            print("No experiments with loss component data found.")
            return
        
        # Sort by best Spearman
        if 'best_spearman' in summary_df.columns:
            summary_df = summary_df.sort_values('best_spearman', ascending=False, na_last=True)
        
        print("\n📊 Loss Component Bounds Analysis\n")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("\n💡 Status Legend:")
        print("   • best: At or below best-case threshold")
        print("   • good: At or below good threshold")
        print("   • acceptable: Between good and poor thresholds")
        print("   • poor: Above poor threshold")
        print("\n📝 See docs/LOSS_COMPONENT_BOUNDS.md for detailed bounds")


if __name__ == '__main__':
    main()

