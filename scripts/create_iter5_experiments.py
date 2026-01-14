#!/usr/bin/env python3
"""
Create Iteration 5 experiment configurations based on Iter4 findings.

Key Findings from Iter4:
- Best: iter4_residual_distillation (0.187) - ModernBERT distillation works!
- Second: iter4_residual_adaptive_reg (0.178) - Adaptive regularization works!
- Most failed (0.0316) - embedding LR 0.1× was too low (fixed to 0.3×)

Iter5 Strategy:
1. Combine distillation + adaptive_reg (best of both)
2. Test with corrected embedding LR (0.3× instead of 0.1×)
3. Fine-tune distillation temperature and alpha
4. Test longer training with best configs
5. Explore other successful combinations
"""

import json
from pathlib import Path

# Base configuration from Iter4 best
iter5_base = {
    'aim_experiment': 'icf-training',
    'use_research_aligned_loss': True,
    'use_unified_loss': False,
    'model_type': 'residual',
    'spearman_weight': 12.0,  # Best from Iter3
    'rank_weight': 0.4,  # Best from Iter3
    'ranking_method': 'probabilistic',  # Best ranking method from Iter3
    'use_spearman': True,
    'use_focal': True,
    'focal_gamma': 2.0,
    'adaptive_reg': True,
    'asymmetry_factor': 2.0,
    'rank_margin': 0.1,
    'batch_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.4,
    'epochs': 200,
    'early_stopping_patience': 15,
    'early_stopping_min_delta': 0.001,
    'use_component_specific_lr': True,  # Fixed: 0.3×/0.5×/1.0×
    'scheduler_type': 'cosine_warmup',
    'clip_grad_norm': 1.0,
}

experiments = []

# Experiment 1: Combine best two (distillation + adaptive_reg)
experiments.append({
    **iter5_base,
    'name': 'iter5_distillation_adaptive_reg',
    'use_distillation': True,
    'teacher_model_name': 'allenai/modernbert-base',
    'teacher_model_type': 'transformers',
    'distillation_temperature': 3.0,
    'distillation_alpha': 0.5,
    'distillation_beta': 0.2,
    'use_feature_distillation': True,
    'batch_size': 32,  # Smaller for ModernBERT
    'description': 'Iter5: Combine distillation (0.187) + adaptive_reg (0.178) - best of both',
})

# Experiment 2: Test with different distillation temperatures
for temp in [2.0, 3.0, 4.0]:
    experiments.append({
        **iter5_base,
        'name': f'iter5_distillation_temp_{int(temp*10)}',
        'use_distillation': True,
        'teacher_model_name': 'allenai/modernbert-base',
        'teacher_model_type': 'transformers',
        'distillation_temperature': temp,
        'distillation_alpha': 0.5,
        'distillation_beta': 0.2,
        'use_feature_distillation': True,
        'batch_size': 32,
        'description': f'Iter5: Distillation with temperature {temp}',
    })

# Experiment 3: Test with different distillation alpha values
for alpha in [0.3, 0.5, 0.7]:
    experiments.append({
        **iter5_base,
        'name': f'iter5_distillation_alpha_{int(alpha*10)}',
        'use_distillation': True,
        'teacher_model_name': 'allenai/modernbert-base',
        'teacher_model_type': 'transformers',
        'distillation_temperature': 3.0,
        'distillation_alpha': alpha,
        'distillation_beta': 0.2,
        'use_feature_distillation': True,
        'batch_size': 32,
        'description': f'Iter5: Distillation with alpha {alpha}',
    })

# Experiment 4: Longer training for best configs
experiments.append({
    **iter5_base,
    'name': 'iter5_distillation_longer',
    'use_distillation': True,
    'teacher_model_name': 'allenai/modernbert-base',
    'teacher_model_type': 'transformers',
    'distillation_temperature': 3.0,
    'distillation_alpha': 0.5,
    'distillation_beta': 0.2,
    'use_feature_distillation': True,
    'batch_size': 32,
    'epochs': 300,
    'early_stopping_patience': 20,
    'description': 'Iter5: Distillation with longer training (300 epochs)',
})

experiments.append({
    **iter5_base,
    'name': 'iter5_adaptive_reg_longer',
    'epochs': 300,
    'early_stopping_patience': 20,
    'description': 'Iter5: Adaptive regularization with longer training (300 epochs)',
})

# Experiment 5: Combine distillation + adaptive_reg with longer training
experiments.append({
    **iter5_base,
    'name': 'iter5_distillation_adaptive_longer',
    'use_distillation': True,
    'teacher_model_name': 'allenai/modernbert-base',
    'teacher_model_type': 'transformers',
    'distillation_temperature': 3.0,
    'distillation_alpha': 0.5,
    'distillation_beta': 0.2,
    'use_feature_distillation': True,
    'adaptive_reg': True,
    'batch_size': 32,
    'epochs': 300,
    'early_stopping_patience': 20,
    'description': 'Iter5: Distillation + adaptive_reg + longer training (300 epochs)',
})

# Experiment 6: Test with different Spearman/ranking ratios around best
for spearman, rank in [(14.0, 0.3), (11.0, 0.5), (13.0, 0.35)]:
    experiments.append({
        **iter5_base,
        'name': f'iter5_ratio_{int(spearman)}x_{int(rank*10)}x',
        'spearman_weight': spearman,
        'rank_weight': rank,
        'description': f'Iter5: Spearman {spearman}× + Ranking {rank}×',
    })

# Experiment 7: Test with higher focal gamma
experiments.append({
    **iter5_base,
    'name': 'iter5_focal_high',
    'focal_gamma': 3.0,
    'description': 'Iter5: Higher focal gamma (3.0)',
})

# Experiment 8: Test with monotonicity constraints
experiments.append({
    **iter5_base,
    'name': 'iter5_monotonicity',
    'use_monotonicity': True,
    'monotonicity_weight': 0.1,
    'description': 'Iter5: Monotonicity constraints',
})

# Experiment 9: Baseline with fixed embedding LR (to verify fix works)
experiments.append({
    **iter5_base,
    'name': 'iter5_baseline_fixed_lr',
    'description': 'Iter5: Baseline with fixed embedding LR (0.3×) to verify fix',
})

print(f"Created {len(experiments)} Iter5 experiment configurations")
print("\nExperiments:")
for exp in experiments:
    print(f"  - {exp['name']}: {exp['description']}")

# Save to JSON for reference
output_path = Path("models/iter5_experiments.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(experiments, f, indent=2)

print(f"\n✅ Saved to {output_path}")

