#!/usr/bin/env python3
"""View diagnostic reports from training experiments."""

import sys
import json
from pathlib import Path
from typing import Optional

def view_diagnostic_report(experiment_dir: str, epoch: Optional[int] = None):
    """View diagnostic report for an experiment."""
    exp_path = Path(experiment_dir)
    
    if not exp_path.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return
    
    # Find latest diagnostic report if epoch not specified
    if epoch is None:
        reports = sorted(exp_path.glob("diagnostic_report_epoch_*.txt"), reverse=True)
        if reports:
            report_path = reports[0]
            epoch = int(report_path.stem.split("_")[-1])
        else:
            print(f"❌ No diagnostic reports found in {experiment_dir}")
            return
    else:
        report_path = exp_path / f"diagnostic_report_epoch_{epoch}.txt"
        if not report_path.exists():
            print(f"❌ Diagnostic report not found: {report_path}")
            return
    
    # Print report
    print(f"📊 Diagnostic Report: {exp_path.name} (Epoch {epoch})")
    print("=" * 70)
    print()
    with open(report_path, 'r') as f:
        print(f.read())
    
    # Also show JSON data if available
    json_path = exp_path / f"diagnostic_data_epoch_{epoch}.json"
    if json_path.exists():
        print("\n💾 JSON data available at:", json_path)
        print("   Use this for programmatic analysis")


def compare_diagnostics(experiment_dirs: list, epoch: Optional[int] = None):
    """Compare diagnostic metrics across experiments."""
    print("📊 Diagnostic Comparison")
    print("=" * 70)
    print()
    
    results = []
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        if epoch is None:
            json_files = sorted(exp_path.glob("diagnostic_data_epoch_*.json"), reverse=True)
            if json_files:
                json_path = json_files[0]
                epoch = int(json_path.stem.split("_")[-1])
            else:
                continue
        else:
            json_path = exp_path / f"diagnostic_data_epoch_{epoch}.json"
            if not json_path.exists():
                continue
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                results.append({
                    'name': exp_path.name,
                    'epoch': epoch,
                    'metrics': data['distances'],
                    'cases': {k: len(v) for k, v in data['interesting_cases'].items()},
                })
        except Exception as e:
            print(f"⚠️  Error loading {json_path}: {e}")
            continue
    
    if not results:
        print("❌ No diagnostic data found")
        return
    
    # Print comparison table
    print(f"Epoch {epoch} Comparison:")
    print()
    print(f"{'Experiment':<30} {'MSE':<10} {'% Close 10%':<12} {'% Close 0.05':<12} {'FP':<6} {'FN':<6} {'Worst':<6}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['metrics']['mean_squared_error']):
        m = r['metrics']
        c = r['cases']
        print(f"{r['name']:<30} {m['mean_squared_error']:<10.4f} "
              f"{m['percent_close_10pct']:<12.1f} {m['percent_close_abs_05']:<12.1f} "
              f"{c.get('false_positives', 0):<6} {c.get('false_negatives', 0):<6} "
              f"{c.get('worst_offenders', 0):<6}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python view_diagnostics.py <experiment_dir> [epoch]")
        print("  python view_diagnostics.py compare <exp1> <exp2> ... [epoch]")
        return
    
    if sys.argv[1] == "compare":
        if len(sys.argv) < 3:
            print("❌ Need at least 2 experiments to compare")
            return
        epoch = int(sys.argv[-1]) if sys.argv[-1].isdigit() else None
        compare_diagnostics(sys.argv[2:-1] if epoch is not None else sys.argv[2:], epoch)
    else:
        epoch = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
        view_diagnostic_report(sys.argv[1], epoch)


if __name__ == '__main__':
    main()

