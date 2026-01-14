#!/usr/bin/env python3
"""Plan Iteration 7 experiments based on Iter6 findings.

Iteration 7 focuses on:
1. Ensemble of top 3-5 models
2. Longer training for transformer models (especially RoBERTa/BERT)
3. Fine-tuning RoBERTa/BERT with best loss config (balanced_hybrid: 10x + 0.5x)
4. Combining distillation + transformer architectures
5. Hyperparameter tuning around best configs
"""

import json
from pathlib import Path

# Base config for transformer fine-tuning (using best loss config)
iter7_transformer_finetune = {
    'aim_experiment': 'icf-training',
    'use_research_aligned_loss': True,
    'use_unified_loss': False,
    'model_type': 'transformer',
    'transformer_model_name': 'roberta-base',
    'transformer_pooling': 'mean',
    'freeze_transformer_backbone': False,
    'use_pretrained_transformer': True,
    # Best loss config from loss_ablation_balanced_hybrid
    'spearman_weight': 10.0,  # From balanced_hybrid (was 12.0 in Iter6)
    'rank_weight': 0.5,  # From balanced_hybrid (was 0.4 in Iter6)
    'ranking_method': 'probabilistic',
    'use_spearman': True,
    'use_focal': True,
    'focal_gamma': 2.0,
    'adaptive_reg': True,
    'asymmetry_factor': 2.0,
    'rank_margin': 0.1,
    'batch_size': 16,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'epochs': 300,  # Longer training
    'early_stopping_patience': 20,
    'early_stopping_min_delta': 0.001,
    'use_component_specific_lr': False,
    'scheduler_type': 'cosine_warmup',
    'clip_grad_norm': 1.0,
}

# Base config for ensemble experiments
iter7_ensemble_base = {
    'aim_experiment': 'icf-training',
    'use_research_aligned_loss': True,
    'use_unified_loss': False,
    'model_type': 'residual',  # Will be ensemble of multiple models
    'spearman_weight': 10.0,
    'rank_weight': 0.5,
    'ranking_method': 'probabilistic',
    'use_spearman': True,
    'use_focal': True,
    'focal_gamma': 2.0,
    'adaptive_reg': True,
    'batch_size': 64,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'epochs': 200,
    'early_stopping_patience': 15,
    'early_stopping_min_delta': 0.001,
    'scheduler_type': 'cosine_warmup',
    'clip_grad_norm': 1.0,
}

experiments = []

# 1. Transformer fine-tuning with best loss config
experiments.append({
    **iter7_transformer_finetune,
    'name': 'iter7_roberta_best_loss',
    'description': 'Iter7: RoBERTa with best loss config (10x + 0.5x), longer training',
})

experiments.append({
    **iter7_transformer_finetune,
    'name': 'iter7_bert_best_loss',
    'transformer_model_name': 'bert-base-uncased',
    'description': 'Iter7: BERT-base with best loss config (10x + 0.5x), longer training',
})

# 2. Transformer + Distillation
experiments.append({
    **iter7_transformer_finetune,
    'name': 'iter7_roberta_distillation',
    'use_distillation': True,
    'distillation_teacher': 'allenai/modernbert-base',
    'distillation_alpha': 0.5,
    'distillation_temperature': 4.0,
    'description': 'Iter7: RoBERTa + ModernBERT distillation with best loss config',
})

# 3. Fine-tune best transformer from Iter6
experiments.append({
    **iter7_transformer_finetune,
    'name': 'iter7_finetune_roberta_iter6',
    'init_from_checkpoint': 'models/iter6_roberta/checkpoint_best.pt',
    'lr': 1e-4,  # Lower LR for fine-tuning
    'epochs': 200,
    'description': 'Iter7: Fine-tune iter6_roberta (0.1845) with best loss config',
})

experiments.append({
    **iter7_transformer_finetune,
    'name': 'iter7_finetune_bert_iter6',
    'transformer_model_name': 'bert-base-uncased',
    'init_from_checkpoint': 'models/iter6_bert_base/checkpoint_best.pt',
    'lr': 1e-4,
    'epochs': 200,
    'description': 'Iter7: Fine-tune iter6_bert_base (0.1837) with best loss config',
})

# 4. Longer training variants
experiments.append({
    **iter7_transformer_finetune,
    'name': 'iter7_roberta_very_long',
    'epochs': 500,
    'early_stopping_patience': 30,
    'description': 'Iter7: RoBERTa with very long training (500 epochs)',
})

# 5. Hyperparameter tuning around best config
experiments.append({
    **iter7_transformer_finetune,
    'name': 'iter7_roberta_tuned_12x_06x',
    'spearman_weight': 12.0,
    'rank_weight': 0.6,
    'description': 'Iter7: RoBERTa with tuned weights (12x + 0.6x)',
})

experiments.append({
    **iter7_transformer_finetune,
    'name': 'iter7_roberta_tuned_8x_04x',
    'spearman_weight': 8.0,
    'rank_weight': 0.4,
    'description': 'Iter7: RoBERTa with tuned weights (8x + 0.4x)',
})

# 6. ResidualICF with best loss config (for comparison)
experiments.append({
    **iter7_ensemble_base,
    'name': 'iter7_residual_best_loss',
    'model_type': 'residual',
    'spearman_weight': 10.0,
    'rank_weight': 0.5,
    'description': 'Iter7: ResidualICF with best loss config (10x + 0.5x)',
})

# 7. DistilBERT with best config (lighter alternative)
experiments.append({
    **iter7_transformer_finetune,
    'name': 'iter7_distilbert_best_loss',
    'transformer_model_name': 'distilbert-base-uncased',
    'batch_size': 32,
    'description': 'Iter7: DistilBERT with best loss config (lighter, faster)',
})

# Save to JSON
output_path = Path('models/iter7_experiments.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(experiments, f, indent=2)

print(f"✅ Planned {len(experiments)} Iteration 7 experiment configurations")
print(f"   Saved to: {output_path}")
print(f"\nExperiments:")
for exp in experiments:
    print(f"  - {exp['name']}: {exp['description']}")

