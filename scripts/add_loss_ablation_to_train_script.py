#!/usr/bin/env python3
"""Add loss ablation experiments to train_flexible_opportunistic.py"""

import re
from pathlib import Path

def add_experiments_to_train_script():
    """Add loss ablation experiments to the training script."""
    
    train_script = Path('../trainctl/training/scripts/train_flexible_opportunistic.py')
    
    if not train_script.exists():
        print(f"❌ Training script not found: {train_script}")
        return
    
    content = train_script.read_text()
    
    # Find where to insert (before the return statement)
    # Look for the last configs.append or before "return configs"
    return_match = re.search(r'(\s+return configs)', content)
    if not return_match:
        print("❌ Could not find 'return configs' in training script")
        return
    
    insert_pos = return_match.start()
    
    # Generate experiment configs
    ablation_configs = """
    # ============================================================================
    # Loss Ablation Experiments (Data Science: Systematic Loss Component Analysis)
    # ============================================================================
    
    # Experiment 1: Pure Spearman (no ranking loss)
    configs.append({
        'name': 'loss_ablation_pure_spearman',
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
        'use_spearman': True,
        'spearman_weight': 10.0,
        'rank_weight': 0.0,  # No ranking loss
        'use_monotonicity': False,
        'use_quantile': False,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,
        'description': 'Pure Spearman optimization (no ranking loss)',
    })
    
    # Experiment 2: Pure Ranking (no Spearman)
    configs.append({
        'name': 'loss_ablation_pure_ranking',
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
        'use_spearman': False,
        'rank_weight': 1.0,  # Higher ranking weight
        'use_monotonicity': False,
        'use_quantile': False,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,
        'description': 'Pure ranking loss (no Spearman)',
    })
    
    # Experiment 3: Balanced Hybrid (current baseline)
    configs.append({
        'name': 'loss_ablation_balanced_hybrid',
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
        'use_spearman': True,
        'spearman_weight': 10.0,
        'rank_weight': 0.5,
        'use_monotonicity': False,
        'use_quantile': False,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,
        'description': 'Balanced hybrid (current: 10× Spearman + 0.5× Ranking)',
    })
    
    # Experiment 4: High Spearman Weight
    configs.append({
        'name': 'loss_ablation_high_spearman',
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
        'use_spearman': True,
        'spearman_weight': 20.0,
        'rank_weight': 0.5,
        'use_monotonicity': False,
        'use_quantile': False,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,
        'description': 'High Spearman weight (20×)',
    })
    
    # Experiment 5: Very High Spearman Weight
    configs.append({
        'name': 'loss_ablation_very_high_spearman',
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
        'use_spearman': True,
        'spearman_weight': 50.0,
        'rank_weight': 0.1,  # Reduced ranking
        'use_monotonicity': False,
        'use_quantile': False,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,
        'description': 'Very high Spearman weight (50×, minimal ranking)',
    })
    
    # Experiment 6: High Ranking Weight
    configs.append({
        'name': 'loss_ablation_high_ranking',
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
        'use_spearman': True,
        'spearman_weight': 5.0,  # Reduced Spearman
        'rank_weight': 2.0,  # High ranking
        'use_monotonicity': False,
        'use_quantile': False,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,
        'description': 'High ranking weight (2×, reduced Spearman)',
    })
    
    # Experiment 7: No Focal Loss
    configs.append({
        'name': 'loss_ablation_no_focal',
        'aim_experiment': 'icf-training',
        'model_type': 'universal',
        'use_research_aligned_loss': True,
        'use_unified_loss': False,
        'adaptive_reg': True,
        'use_focal': False,  # No focal weighting
        'focal_gamma': 2.0,
        'asymmetry_factor': 2.0,
        'rank_margin': 0.1,
        'ranking_method': 'sigmoid',
        'use_spearman': True,
        'spearman_weight': 10.0,
        'rank_weight': 0.5,
        'use_monotonicity': False,
        'use_quantile': False,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,
        'description': 'No focal loss (baseline hybrid without focal)',
    })
    
    # Experiment 8: With Monotonicity
    configs.append({
        'name': 'loss_ablation_with_monotonicity',
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
        'use_spearman': True,
        'spearman_weight': 10.0,
        'rank_weight': 0.5,
        'use_monotonicity': True,
        'monotonicity_weight': 0.1,
        'use_quantile': False,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,
        'description': 'Hybrid + monotonicity constraints',
    })
    
    # Experiment 9: Low Spearman Weight
    configs.append({
        'name': 'loss_ablation_low_spearman',
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
        'use_spearman': True,
        'spearman_weight': 5.0,
        'rank_weight': 1.0,
        'use_monotonicity': False,
        'use_quantile': False,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,
        'description': 'Low Spearman weight (5×, higher ranking)',
    })
    
    # Experiment 10: Equal Weights
    configs.append({
        'name': 'loss_ablation_equal_weights',
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
        'use_spearman': True,
        'spearman_weight': 1.0,
        'rank_weight': 1.0,
        'use_monotonicity': False,
        'use_quantile': False,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'epochs': 100,
        'description': 'Equal weights (1× Spearman + 1× Ranking)',
    })
    
"""
    
    # Insert before return statement
    new_content = content[:insert_pos] + ablation_configs + content[insert_pos:]
    
    # Write back
    train_script.write_text(new_content)
    print(f"✅ Added {10} loss ablation experiments to {train_script}")
    print("   Experiments will be available when training script is run")


if __name__ == '__main__':
    add_experiments_to_train_script()

