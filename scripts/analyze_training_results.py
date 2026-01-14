# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
#   "json5>=0.9.0",
# ]
# ///
"""
Analyze training results from flexible opportunistic training.

Compares experiments, identifies best configurations, and provides insights.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_experiment_summary(summary_path: Path) -> List[Dict]:
    """Load experiment summary JSON."""
    if not summary_path.exists():
        return []
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def analyze_results(summary_path: Path = Path("models/experiment_summary.json")):
    """Analyze and compare experiment results."""
    print("=" * 70)
    print("TRAINING RESULTS ANALYSIS")
    print("=" * 70)
    
    results = load_experiment_summary(summary_path)
    
    if not results:
        print("No results found. Training may still be in progress.")
        return
    
    # Filter successful experiments
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\n📊 Summary: {len(successful)} successful, {len(failed)} failed")
    
    if failed:
        print("\n❌ Failed Experiments:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")
    
    if successful:
        print("\n✅ Successful Experiments:")
        print(f"{'Experiment':<25} {'Best Spearman':<15} {'Final Spearman':<15} {'Final MAE':<12} {'Epochs':<8}")
        print("-" * 75)
        
        for r in successful:
            print(f"{r['name']:<25} "
                  f"{r.get('best_spearman', 0):.4f}        "
                  f"{r.get('final_spearman', 0):.4f}        "
                  f"{r.get('final_mae', 0):.4f}      "
                  f"{r.get('epochs_trained', 0)}")
        
        # Find best
        best = max(successful, key=lambda x: x.get('best_spearman', -1))
        print(f"\n🏆 Best Experiment: {best['name']}")
        print(f"   Best Spearman: {best.get('best_spearman', 0):.4f}")
        print(f"   Final Spearman: {best.get('final_spearman', 0):.4f}")
        print(f"   Final MAE: {best.get('final_mae', 0):.4f}")
        
        # Compare configurations
        print("\n📈 Configuration Comparison:")
        for r in successful:
            print(f"\n  {r['name']}:")
            print(f"    Spearman: {r.get('best_spearman', 0):.4f}")
            print(f"    MAE: {r.get('final_mae', 0):.4f}")
            print(f"    Epochs: {r.get('epochs_trained', 0)}")
    
    # Check for history files with loss component analysis
    print("\n🔍 Loss Component Analysis:")
    for result in successful:
        exp_name = result['name']
        history_path = Path("models") / exp_name / "history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            if 'loss_components' in history and history['loss_components']:
                # Get average ratios
                components = history['loss_components']
                avg_ratios = {}
                for comp in components:
                    for k, v in comp.items():
                        avg_ratios[k] = avg_ratios.get(k, 0) + v
                
                n = len(components)
                if n > 0:
                    print(f"\n  {exp_name}:")
                    for k, v in avg_ratios.items():
                        print(f"    {k}: {v/n:.2%}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("models/experiment_summary.json"))
    args = parser.parse_args()
    
    analyze_results(args.summary)


