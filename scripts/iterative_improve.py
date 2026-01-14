#!/usr/bin/env python3
"""Iteratively improve experiments based on monitoring results."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

PROJECT_ROOT = Path(__file__).parent.parent


def load_latest_metrics(exp_name: str) -> Optional[Dict]:
    """Load latest metrics from an experiment."""
    metrics_csv = PROJECT_ROOT / "models" / exp_name / "lightning_logs" / "version_0" / "metrics.csv"
    
    if not metrics_csv.exists():
        return None
    
    try:
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        
        # Get latest validation metrics
        val_df = df[df['val_spearman_corr'].notna()]
        if len(val_df) == 0:
            return None
        
        latest = val_df.iloc[-1]
        return {
            'epoch': int(latest.get('epoch', 0)),
            'val_spearman': float(latest.get('val_spearman_corr', 0)),
            'val_mae': float(latest.get('val_mae', 0)),
            'train_loss': float(latest.get('train_loss', 0)),
            'val_loss': float(latest.get('val_loss', 0)),
        }
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None


def analyze_training_status(exp_name: str) -> Dict:
    """Analyze training status and suggest improvements."""
    metrics = load_latest_metrics(exp_name)
    
    if metrics is None:
        return {
            'status': 'not_started',
            'suggestions': ['Wait for training to start']
        }
    
    suggestions = []
    status = 'training'
    
    # Check for overfitting
    if metrics['val_loss'] > metrics['train_loss'] * 1.2:
        suggestions.append("⚠️  Overfitting detected: Increase dropout or weight_decay")
    
    # Check for slow convergence
    if metrics['epoch'] > 20 and metrics['val_spearman'] < 0.1:
        suggestions.append("⚠️  Slow convergence: Consider increasing learning rate or adjusting loss weights")
    
    # Check for plateau
    if metrics['epoch'] > 30:
        # Would need historical data to detect plateau
        suggestions.append("💡 Consider early stopping if no improvement")
    
    # Check for good progress
    if metrics['val_spearman'] > 0.2:
        suggestions.append("✅ Good progress! Consider continuing training")
    
    return {
        'status': status,
        'metrics': metrics,
        'suggestions': suggestions,
    }


def suggest_config_improvements(exp_name: str, analysis: Dict) -> List[Dict]:
    """Suggest configuration improvements based on analysis."""
    improvements = []
    
    if not analysis.get('metrics'):
        return improvements
    
    metrics = analysis['metrics']
    
    # If overfitting, suggest stronger regularization
    if any('Overfitting' in s for s in analysis.get('suggestions', [])):
        improvements.append({
            'type': 'regularization',
            'suggestion': 'Increase dropout to 0.4 and weight_decay to 2e-4',
            'config_changes': {
                'dropout': 0.4,
                'weight_decay': 2e-4,
            }
        })
    
    # If slow convergence, suggest learning rate adjustment
    if any('Slow convergence' in s for s in analysis.get('suggestions', [])):
        improvements.append({
            'type': 'learning_rate',
            'suggestion': 'Try higher learning rate (2e-3) or adjust loss weights',
            'config_changes': {
                'lr': 2e-3,
            }
        })
    
    # If good progress, suggest continuing or trying advanced features
    if metrics['val_spearman'] > 0.2:
        improvements.append({
            'type': 'advanced_features',
            'suggestion': 'Consider enabling monotonicity constraints or quantile regression',
            'config_changes': {
                'use_monotonicity': True,
                'monotonicity_weight': 0.1,
            }
        })
    
    return improvements


def main():
    """Main entry point."""
    exp_name = sys.argv[1] if len(sys.argv) > 1 else 'research_aligned_standard'
    
    print(f"🔍 Analyzing: {exp_name}\n")
    
    analysis = analyze_training_status(exp_name)
    
    print(f"Status: {analysis['status']}")
    if analysis.get('metrics'):
        m = analysis['metrics']
        print(f"Epoch: {m['epoch']}")
        print(f"Val Spearman: {m['val_spearman']:.4f}")
        print(f"Val MAE: {m['val_mae']:.4f}")
        print(f"Train Loss: {m['train_loss']:.4f}")
        print(f"Val Loss: {m['val_loss']:.4f}")
    
    print("\n💡 Suggestions:")
    for suggestion in analysis.get('suggestions', []):
        print(f"   {suggestion}")
    
    improvements = suggest_config_improvements(exp_name, analysis)
    if improvements:
        print("\n🔧 Suggested Improvements:")
        for imp in improvements:
            print(f"   {imp['type']}: {imp['suggestion']}")
            print(f"      Config: {imp['config_changes']}")


if __name__ == '__main__':
    main()

