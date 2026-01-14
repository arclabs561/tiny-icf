#!/usr/bin/env python3
"""Analyze loss component breakdown from experiment metrics.

This script helps verify that loss component logging is working correctly
and provides insights into which components are driving improvements.
"""

import sys
from pathlib import Path
import csv
import json
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_metrics_csv(exp_dir: Path) -> Optional[List[Dict]]:
    """Load metrics CSV from experiment directory."""
    lightning_logs = exp_dir / "lightning_logs"
    if not lightning_logs.exists():
        return None
    
    # Find latest version
    version_dirs = sorted(lightning_logs.glob("version_*"), reverse=True)
    for version_dir in version_dirs:
        metrics_csv = version_dir / "metrics.csv"
        if metrics_csv.exists():
            try:
                with open(metrics_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            except Exception as e:
                print(f"Error loading {metrics_csv}: {e}")
                continue
    
    return None


def extract_loss_components(metrics: List[Dict]) -> Dict[str, List[float]]:
    """Extract loss component values from metrics."""
    components = {}
    
    # Find all loss component columns
    if not metrics:
        return components
    
    for key in metrics[0].keys():
        if key.startswith('val_loss_') or key.startswith('train_loss_'):
            component_name = key.replace('val_loss_', '').replace('train_loss_', '')
            if component_name not in ['total', 'step', 'epoch']:
                values = []
                for row in metrics:
                    val = row.get(key, '')
                    if val and val not in ['', 'nan', 'NaN']:
                        try:
                            values.append(float(val))
                        except (ValueError, TypeError):
                            pass
                if values:
                    components[component_name] = values
    
    return components


def analyze_experiment(exp_name: str, base_dir: Path) -> Optional[Dict]:
    """Analyze a single experiment."""
    exp_dir = base_dir / exp_name
    if not exp_dir.exists():
        return None
    
    metrics = load_metrics_csv(exp_dir)
    if not metrics:
        return None
    
    # Get validation metrics
    val_metrics = [r for r in metrics if r.get('val_spearman_corr') and r['val_spearman_corr'] not in ['', 'nan', 'NaN']]
    if not val_metrics:
        return None
    
    # Get best Spearman
    spearmans = [float(r['val_spearman_corr']) for r in val_metrics]
    best_spearman = max(spearmans)
    latest_spearman = spearmans[-1] if spearmans else 0.0
    
    # Extract loss components
    loss_components = extract_loss_components(metrics)
    
    # Get latest loss component values
    latest_components = {}
    if val_metrics:
        latest_row = val_metrics[-1]
        for key in latest_row.keys():
            if key.startswith('val_loss_'):
                component_name = key.replace('val_loss_', '')
                if component_name not in ['total', 'step', 'epoch']:
                    val = latest_row.get(key, '')
                    if val and val not in ['', 'nan', 'NaN']:
                        try:
                            latest_components[component_name] = float(val)
                        except (ValueError, TypeError):
                            pass
    
    return {
        'name': exp_name,
        'best_spearman': best_spearman,
        'latest_spearman': latest_spearman,
        'epochs': len(val_metrics),
        'loss_components': latest_components,
        'all_components': loss_components,
    }


def main():
    """Main analysis function."""
    base_dir = project_root / "models"
    
    # Analyze top experiments
    top_experiments = [
        'loss_ablation_balanced_hybrid',
        'iter4_residual_distillation',
        'residual_balanced',
        'iter6_roberta',
        'iter6_bert_base',
    ]
    
    print("📊 Loss Component Analysis\n")
    print("=" * 80)
    print()
    
    results = []
    for exp_name in top_experiments:
        analysis = analyze_experiment(exp_name, base_dir)
        if analysis:
            results.append(analysis)
    
    # Print results
    print(f"{'Experiment':<45} {'Best Spearman':>15} {'Loss Components':<30}")
    print("-" * 95)
    
    for result in results:
        name = result['name'][:43]
        spearman = result['best_spearman']
        components = result['loss_components']
        
        # Format components
        if components:
            comp_str = ", ".join([f"{k}={v:.4f}" for k, v in list(components.items())[:3]])
            if len(components) > 3:
                comp_str += "..."
        else:
            comp_str = "No components logged"
        
        print(f"{name:<45} {spearman:>15.4f} {comp_str:<30}")
    
    print()
    print("💡 Note: Loss components are logged during validation")
    print("   Check metrics.csv for val_loss_* columns")
    print("   If missing, ensure ResearchAlignedICFLoss is being used")


if __name__ == '__main__':
    main()

