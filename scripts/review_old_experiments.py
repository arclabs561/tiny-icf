#!/usr/bin/env python3
"""
Comprehensive review of old experiments to identify patterns and insights.
"""

import json
from pathlib import Path
from collections import defaultdict
import csv

def review_experiments():
    """Review all past experiments comprehensively."""
    print("📊 Comprehensive Experiment Review")
    print("=" * 80)
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return
    
    # Collect experiment data
    experiments = []
    
    # Check comprehensive analysis file
    analysis_path = models_dir / "comprehensive_analysis.json"
    if analysis_path.exists():
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
            experiments.extend(analysis.get('experiments', []))
    
    # Also check individual experiment directories
    for exp_dir in models_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue
        
        # Try to read metrics from CSV
        csv_path = exp_dir / "lightning_logs" / "version_0" / "metrics.csv"
        if csv_path.exists():
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows and 'val_spearman_corr' in rows[0]:
                        val_spearman = [float(r['val_spearman_corr']) for r in rows if r.get('val_spearman_corr') and r['val_spearman_corr'].strip()]
                        if val_spearman:
                            best = max(val_spearman)
                            final = val_spearman[-1]
                            
                            experiments.append({
                                'name': exp_dir.name,
                                'best_val_spearman_corr': float(best),
                                'final_val_spearman_corr': float(final),
                                'epochs_trained': len(val_spearman),
                            })
            except Exception as e:
                pass
    
    if not experiments:
        print("⚠️  No experiment data found")
        return
    
    # Sort by best Spearman
    sorted_exps = sorted(experiments, 
                        key=lambda x: x.get('best_val_spearman_corr', -1), 
                        reverse=True)
    
    print(f"\n✅ Found {len(sorted_exps)} experiments")
    
    # Top performers
    print("\n🏆 Top 15 Experiments by Best Spearman:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Experiment Name':<45} {'Best':<8} {'Final':<8} {'Epochs':<8}")
    print("-" * 80)
    for i, exp in enumerate(sorted_exps[:15], 1):
        name = exp.get('name', 'unknown')
        best = exp.get('best_val_spearman_corr', 0)
        final = exp.get('final_val_spearman_corr', 0)
        epochs = exp.get('epochs_trained', 0)
        print(f"{i:<6} {name:<45} {best:<8.4f} {final:<8.4f} {epochs:<8}")
    
    # Analyze by category
    print("\n📈 Performance by Category:")
    print("-" * 80)
    categories = defaultdict(list)
    
    for exp in sorted_exps:
        name = exp.get('name', '')
        best = exp.get('best_val_spearman_corr', 0)
        
        # Categorize
        if 'iter5' in name:
            cat = 'Iteration 5'
        elif 'iter4' in name:
            cat = 'Iteration 4'
        elif 'iter3' in name:
            cat = 'Iteration 3'
        elif 'loss_ablation' in name:
            cat = 'Loss Ablation'
        elif 'distillation' in name:
            cat = 'Distillation'
        elif 'residual' in name:
            cat = 'Residual Architecture'
        elif 'research_aligned' in name:
            cat = 'Research Aligned'
        else:
            cat = 'Other'
        
        categories[cat].append(best)
    
    print(f"{'Category':<25} {'Count':<8} {'Avg':<10} {'Max':<10} {'Min':<10}")
    print("-" * 80)
    for cat in sorted(categories.keys(), key=lambda x: max(categories[x]) if categories[x] else 0, reverse=True):
        scores = categories[cat]
        if scores:
            avg = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            print(f"{cat:<25} {len(scores):<8} {avg:<10.4f} {max_score:<10.4f} {min_score:<10.4f}")
    
    # Key insights
    print("\n💡 Key Insights:")
    print("-" * 80)
    
    # Find best in each category
    best_by_cat = {}
    for cat, scores in categories.items():
        if scores:
            best_by_cat[cat] = max(scores)
    
    if best_by_cat:
        best_overall = max(best_by_cat.values())
        best_cats = [cat for cat, score in best_by_cat.items() if score == best_overall]
        if best_cats:
            print(f"  🏆 Best overall: {best_overall:.4f} ({best_cats[0]})")
    
    # Check for distillation experiments
    distillation_exps = [e for e in sorted_exps if 'distillation' in e.get('name', '').lower()]
    if distillation_exps:
        best_dist = max(e.get('best_val_spearman_corr', 0) for e in distillation_exps)
        print(f"  🎓 Best distillation: {best_dist:.4f} ({len(distillation_exps)} experiments)")
    
    # Check for adaptive_reg experiments
    adaptive_exps = [e for e in sorted_exps if 'adaptive' in e.get('name', '').lower()]
    if adaptive_exps:
        best_adaptive = max(e.get('best_val_spearman_corr', 0) for e in adaptive_exps)
        print(f"  📊 Best adaptive_reg: {best_adaptive:.4f} ({len(adaptive_exps)} experiments)")
    
    # Check for residual architecture
    residual_exps = [e for e in sorted_exps if 'residual' in e.get('name', '').lower()]
    if residual_exps:
        best_residual = max(e.get('best_val_spearman_corr', 0) for e in residual_exps)
        print(f"  🏗️  Best residual: {best_residual:.4f} ({len(residual_exps)} experiments)")
    
    # Recommendations
    print("\n🎯 Recommendations for Next Iteration:")
    print("-" * 80)
    
    if best_by_cat.get('Distillation', 0) > 0.18:
        print("  ✅ Distillation shows strong results - continue exploring")
        print("     - Test different temperatures (2.0, 3.0, 4.0)")
        print("     - Test different alpha values (0.3, 0.5, 0.7)")
        print("     - Combine with adaptive regularization")
    
    if best_by_cat.get('Iteration 4', 0) > 0.17:
        print("  ✅ Iter4 shows improvement - build on best configs")
        print("     - Combine distillation + adaptive_reg")
        print("     - Test longer training (300 epochs)")
    
    if best_by_cat.get('Residual Architecture', 0) > 0.18:
        print("  ✅ Residual architecture performs well - continue using")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    review_experiments()

