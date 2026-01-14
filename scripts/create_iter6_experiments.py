#!/usr/bin/env python3
"""Create Iteration 6 experiment configurations.

Iteration 6 focuses on:
1. Transformer architectures (DistilBERT, ByT5, BERT-base, RoBERTa)
2. Fine-tuning from best checkpoints (residual_balanced, iter4_residual_distillation, etc.)
3. Character-level transformers for better ICF prediction
"""

import json
from pathlib import Path

# Base config for transformer experiments
iter6_transformer_base = {
    'aim_experiment': 'icf-training',
    'use_research_aligned_loss': True,
    'use_unified_loss': False,
    'model_type': 'transformer',
    'transformer_model_name': 'distilbert-base-uncased',
    'transformer_pooling': 'mean',
    'freeze_transformer_backbone': False,
    'use_pretrained_transformer': True,
    'spearman_weight': 12.0,
    'rank_weight': 0.4,
    'ranking_method': 'probabilistic',
    'use_spearman': True,
    'use_focal': True,
    'focal_gamma': 2.0,
    'adaptive_reg': True,
    'asymmetry_factor': 2.0,
    'rank_margin': 0.1,
    'batch_size': 32,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'epochs': 200,
    'early_stopping_patience': 15,
    'early_stopping_min_delta': 0.001,
    'use_component_specific_lr': False,
    'scheduler_type': 'cosine_warmup',
    'clip_grad_norm': 1.0,
}

# Base config for fine-tuning experiments (CNN-based)
iter6_finetune_base = {
    'aim_experiment': 'icf-training',
    'use_research_aligned_loss': True,
    'use_unified_loss': False,
    'model_type': 'residual',
    'spearman_weight': 12.0,
    'rank_weight': 0.4,
    'ranking_method': 'probabilistic',
    'use_spearman': True,
    'use_focal': True,
    'focal_gamma': 2.0,
    'adaptive_reg': True,
    'asymmetry_factor': 2.0,
    'rank_margin': 0.1,
    'batch_size': 64,
    'lr': 1e-4,  # Lower LR for fine-tuning
    'weight_decay': 1e-4,
    'dropout': 0.4,
    'epochs': 100,  # Shorter for fine-tuning
    'early_stopping_patience': 15,
    'early_stopping_min_delta': 0.001,
    'use_component_specific_lr': True,
    'scheduler_type': 'cosine_warmup',
    'clip_grad_norm': 1.0,
}

experiments = []

# Transformer architecture experiments
experiments.append({
    **iter6_transformer_base,
    'name': 'iter6_distilbert',
    'description': 'Iter6: DistilBERT transformer architecture (lightweight, efficient)',
})

experiments.append({
    **iter6_transformer_base,
    'name': 'iter6_byt5',
    'model_type': 'char_transformer',
    'transformer_model_name': 'google/byt5-small',
    'description': 'Iter6: ByT5 character-level transformer (byte-level processing)',
})

experiments.append({
    **iter6_transformer_base,
    'name': 'iter6_bert_base',
    'transformer_model_name': 'bert-base-uncased',
    'batch_size': 16,
    'description': 'Iter6: BERT-base transformer architecture (larger capacity)',
})

experiments.append({
    **iter6_transformer_base,
    'name': 'iter6_roberta',
    'transformer_model_name': 'roberta-base',
    'batch_size': 16,
    'description': 'Iter6: RoBERTa-base transformer architecture (improved BERT)',
})

experiments.append({
    **iter6_transformer_base,
    'name': 'iter6_distilbert_frozen',
    'freeze_transformer_backbone': True,
    'lr': 1e-3,
    'description': 'Iter6: DistilBERT with frozen backbone (feature extraction only)',
})

experiments.append({
    **iter6_transformer_base,
    'name': 'iter6_distilbert_learned_pooling',
    'transformer_pooling': 'learned',
    'description': 'Iter6: DistilBERT with learned attention pooling',
})

experiments.append({
    **iter6_transformer_base,
    'name': 'iter6_byt5_longer',
    'model_type': 'char_transformer',
    'transformer_model_name': 'google/byt5-small',
    'epochs': 300,
    'early_stopping_patience': 20,
    'description': 'Iter6: ByT5 with longer training (300 epochs)',
})

# Fine-tuning from best checkpoints
experiments.append({
    **iter6_finetune_base,
    'name': 'iter6_finetune_residual_balanced',
    'init_from_checkpoint': 'models/residual_balanced/checkpoint_best.pt',
    'description': 'Iter6: Fine-tune residual_balanced (0.1864 Spearman) with new loss/config',
})

experiments.append({
    **iter6_finetune_base,
    'name': 'iter6_finetune_distillation',
    'init_from_checkpoint': 'models/iter4_residual_distillation/checkpoint_best.pt',
    'description': 'Iter6: Fine-tune iter4_residual_distillation (0.1875 Spearman)',
})

experiments.append({
    **iter6_finetune_base,
    'name': 'iter6_finetune_balanced_hybrid',
    'init_from_checkpoint': 'models/loss_ablation_balanced_hybrid/checkpoint_best.pt',
    'description': 'Iter6: Fine-tune loss_ablation_balanced_hybrid (0.1891 Spearman)',
})

# Save to JSON
output_path = Path('models/iter6_experiments.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(experiments, f, indent=2)

print(f"✅ Created {len(experiments)} Iteration 6 experiment configurations")
print(f"   Saved to: {output_path}")
print(f"\nExperiments:")
for exp in experiments:
    print(f"  - {exp['name']}: {exp['description']}")

