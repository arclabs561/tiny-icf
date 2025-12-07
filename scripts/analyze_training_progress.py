# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
#   "matplotlib>=3.7.0",
# ]
# ///
"""Analyze and visualize training progress from history files."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_history(history_path: Path) -> Optional[Dict]:
    """Load training history."""
    if not history_path.exists():
        return None
    try:
        return json.load(open(history_path))
    except Exception as e:
        print(f"Error loading {history_path}: {e}")
        return None

def analyze_experiment(name: str, history: Dict) -> Dict:
    """Analyze a single experiment."""
    train_spearman = [e.get('spearman_corr', e.get('spearman', 0)) for e in history.get('train', [])]
    val_spearman = [e.get('spearman_corr', e.get('spearman', 0)) for e in history.get('val', [])]
    
    if not train_spearman or not val_spearman:
        return None
    
    gaps = [t - v for t, v in zip(train_spearman, val_spearman)]
    best_val_idx = val_spearman.index(max(val_spearman))
    
    return {
        'name': name,
        'epochs': len(train_spearman),
        'best_val': max(val_spearman),
        'best_val_epoch': best_val_idx + 1,
        'final_train': train_spearman[-1],
        'final_val': val_spearman[-1],
        'final_gap': gaps[-1],
        'avg_gap': np.mean(gaps),
        'min_gap': min(gaps),
        'max_gap': max(gaps),
        'gap_at_best': gaps[best_val_idx],
        'train_spearman': train_spearman,
        'val_spearman': val_spearman,
        'gaps': gaps,
    }

def plot_training_curves(experiments: List[Dict], output_path: Path):
    """Plot training curves for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Spearman correlation over time
    ax = axes[0, 0]
    for exp in experiments:
        epochs = range(1, len(exp['train_spearman']) + 1)
        ax.plot(epochs, exp['train_spearman'], label=f"{exp['name']} (train)", linestyle='--', alpha=0.7)
        ax.plot(epochs, exp['val_spearman'], label=f"{exp['name']} (val)", linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gap over time
    ax = axes[0, 1]
    for exp in experiments:
        epochs = range(1, len(exp['gaps']) + 1)
        ax.plot(epochs, exp['gaps'], label=exp['name'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train - Val Gap')
    ax.set_title('Overfitting Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Best validation comparison
    ax = axes[1, 0]
    names = [exp['name'] for exp in experiments]
    best_vals = [exp['best_val'] for exp in experiments]
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
    bars = ax.bar(names, best_vals, color=colors)
    ax.set_ylabel('Best Validation Spearman')
    ax.set_title('Best Performance Comparison')
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, best_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Final gap comparison
    ax = axes[1, 1]
    final_gaps = [exp['final_gap'] for exp in experiments]
    bars = ax.bar(names, final_gaps, color=colors)
    ax.set_ylabel('Final Train-Val Gap')
    ax.set_title('Overfitting Comparison')
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, gap in zip(bars, final_gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{gap:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved training analysis plot to {output_path}")

def main():
    experiments_data = [
        ('temporal_amoo', Path('models/history_temporal_amoo.json')),
        ('reduced_capacity', Path('models/history_reduced_capacity.json')),
        ('batchnorm', Path('models/history_batchnorm.json')),
    ]
    
    experiments = []
    for name, path in experiments_data:
        history = load_history(path)
        if history:
            analysis = analyze_experiment(name, history)
            if analysis:
                experiments.append(analysis)
    
    if not experiments:
        print("No completed experiments found for analysis.")
        return
    
    print("=" * 80)
    print("TRAINING PROGRESS ANALYSIS")
    print("=" * 80)
    print()
    
    # Summary table
    print(f"{'Experiment':<25} {'Epochs':<8} {'Best Val':<10} {'Final Gap':<12} {'Gap %':<8}")
    print("-" * 80)
    for exp in experiments:
        gap_pct = (exp['final_gap'] / exp['final_train'] * 100) if exp['final_train'] > 0 else 0
        print(f"{exp['name']:<25} {exp['epochs']:<8} {exp['best_val']:<10.4f} {exp['final_gap']:<12.4f} {gap_pct:<8.1f}%")
    
    # Detailed analysis
    print()
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    for exp in experiments:
        print(f"\n{exp['name'].upper()}:")
        print(f"  Best Validation: {exp['best_val']:.4f} (epoch {exp['best_val_epoch']})")
        print(f"  Final Train: {exp['final_train']:.4f}")
        print(f"  Final Val: {exp['final_val']:.4f}")
        print(f"  Final Gap: {exp['final_gap']:.4f} ({(exp['final_gap']/exp['final_train']*100):.1f}%)")
        print(f"  Average Gap: {exp['avg_gap']:.4f}")
        print(f"  Gap Range: [{exp['min_gap']:.4f}, {exp['max_gap']:.4f}]")
        print(f"  Gap at Best: {exp['gap_at_best']:.4f}")
    
    # Plot if we have data
    if len(experiments) > 0:
        plot_training_curves(experiments, Path('models/training_analysis.png'))

if __name__ == "__main__":
    main()
