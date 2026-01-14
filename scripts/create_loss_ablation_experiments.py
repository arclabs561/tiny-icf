#!/usr/bin/env python3
"""Create systematic loss ablation experiments to determine optimal configuration."""

import json
from pathlib import Path

def create_loss_ablation_configs():
    """Create systematic experiments varying loss components."""
    
    experiments = []
    
    # Base configuration (from research_aligned_residual)
    base_config = {
        'aim_experiment': 'icf-training',
        'model_type': 'universal',
        'use_research_aligned_loss': True,
        'use_unified_loss': False,
        'adaptive_reg': True,
        'use_focal': True,
        'focal_gamma': 2.0,
        'asymmetry_factor': 2.0,
        'rank_margin': 0.1,
        'ranking_method': 'sigmoid',
        'use_monotonicity': False,
        'use_quantile': False,
        'batch_size': 64,  # CPU-friendly
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,  # Shorter for faster iteration
    }
    
    # Experiment 1: Pure Spearman (no ranking loss)
    experiments.append({
        **base_config,
        'name': 'loss_ablation_pure_spearman',
        'use_spearman': True,
        'spearman_weight': 10.0,
        'rank_weight': 0.0,  # No ranking loss
        'description': 'Pure Spearman optimization (no ranking loss)',
    })
    
    # Experiment 2: Pure Ranking (no Spearman)
    experiments.append({
        **base_config,
        'name': 'loss_ablation_pure_ranking',
        'use_spearman': False,
        'rank_weight': 1.0,  # Higher ranking weight
        'description': 'Pure ranking loss (no Spearman)',
    })
    
    # Experiment 3: Balanced Hybrid (current)
    experiments.append({
        **base_config,
        'name': 'loss_ablation_balanced_hybrid',
        'use_spearman': True,
        'spearman_weight': 10.0,
        'rank_weight': 0.5,
        'description': 'Balanced hybrid (current: 10× Spearman + 0.5× Ranking)',
    })
    
    # Experiment 4: High Spearman Weight
    experiments.append({
        **base_config,
        'name': 'loss_ablation_high_spearman',
        'use_spearman': True,
        'spearman_weight': 20.0,
        'rank_weight': 0.5,
        'description': 'High Spearman weight (20×)',
    })
    
    # Experiment 5: Very High Spearman Weight
    experiments.append({
        **base_config,
        'name': 'loss_ablation_very_high_spearman',
        'use_spearman': True,
        'spearman_weight': 50.0,
        'rank_weight': 0.1,  # Reduced ranking
        'description': 'Very high Spearman weight (50×, minimal ranking)',
    })
    
    # Experiment 6: High Ranking Weight
    experiments.append({
        **base_config,
        'name': 'loss_ablation_high_ranking',
        'use_spearman': True,
        'spearman_weight': 5.0,  # Reduced Spearman
        'rank_weight': 2.0,  # High ranking
        'description': 'High ranking weight (2×, reduced Spearman)',
    })
    
    # Experiment 7: No Focal Loss
    experiments.append({
        **base_config,
        'name': 'loss_ablation_no_focal',
        'use_spearman': True,
        'spearman_weight': 10.0,
        'rank_weight': 0.5,
        'use_focal': False,  # No focal weighting
        'description': 'No focal loss (baseline hybrid without focal)',
    })
    
    # Experiment 8: With Monotonicity
    experiments.append({
        **base_config,
        'name': 'loss_ablation_with_monotonicity',
        'use_spearman': True,
        'spearman_weight': 10.0,
        'rank_weight': 0.5,
        'use_monotonicity': True,
        'monotonicity_weight': 0.1,
        'description': 'Hybrid + monotonicity constraints',
    })
    
    # Experiment 9: Low Spearman Weight
    experiments.append({
        **base_config,
        'name': 'loss_ablation_low_spearman',
        'use_spearman': True,
        'spearman_weight': 5.0,
        'rank_weight': 1.0,
        'description': 'Low Spearman weight (5×, higher ranking)',
    })
    
    # Experiment 10: Equal Weights
    experiments.append({
        **base_config,
        'name': 'loss_ablation_equal_weights',
        'use_spearman': True,
        'spearman_weight': 1.0,
        'rank_weight': 1.0,
        'description': 'Equal weights (1× Spearman + 1× Ranking)',
    })
    
    return experiments


def main():
    """Generate experiment configurations."""
    experiments = create_loss_ablation_configs()
    
    # Save to JSON for reference
    output_path = Path('models/loss_ablation_experiments.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print(f"✅ Created {len(experiments)} loss ablation experiments")
    print(f"💾 Saved to: {output_path}")
    print()
    print("Experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i:2d}. {exp['name']}: {exp['description']}")
    
    return experiments


if __name__ == '__main__':
    main()

