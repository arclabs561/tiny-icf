# Training Refinement Recommendations

## Current Status (Epoch 21/150)

- **Best Spearman:** 0.1706 (epoch 8)
- **Current Spearman:** 0.1613 (epoch 21)
- **Decline:** 0.0093 (5.4% drop from best)
- **Status:** MODERATE performance, showing signs of overfitting

## Key Findings

1. **Performance Decline:** Model peaked at epoch 8, then declined
2. **Overfitting Risk:** Gap between train and validation metrics likely increasing
3. **Moderate Correlation:** 0.17 Spearman is reasonable but has room for improvement

## Refinement Recommendations

### 1. Loss Function Improvements

**Current Configuration:**
- `spearman_weight`: 10.0
- `spearman_reg_strength`: 1.0
- `ranking_reg_strength`: 1.0

**Recommendations:**
- **Increase `spearman_weight` to 20.0-30.0**: Stronger emphasis on ranking order
- **Experiment with `spearman_reg_strength` (0.5-2.0)**: Balance smoothness vs accuracy
- **Increase pairwise ranking weight**: Enforce relative ordering more strongly

### 2. Learning Rate Scheduling

**Current:** Fixed LR or basic scheduler

**Recommendations:**
- **Enable ReduceLROnPlateau**: Monitor `val_spearman_corr`, reduce LR when plateau
- **Use Cosine Annealing with Warmup**: Already available, should be enabled
- **Component-specific LRs**: Already available, tune for embeddings vs MLP

### 3. Early Stopping Configuration

**Need to Verify:**
- Monitor: `val_spearman_corr`
- Patience: 10-15 epochs
- Min delta: 0.001 (0.1% improvement threshold)
- Mode: `max` (higher is better)

### 4. Regularization Adjustments

**Current:**
- `dropout`: 0.4
- `weight_decay`: 1e-4

**Recommendations:**
- **Increase dropout to 0.5** if overfitting continues
- **Gradient clipping**: Already available, ensure it's enabled (clip_grad_norm=1.0)
- **Label smoothing**: For classification tasks in multi-task setup

### 5. Multi-Task Learning

**Current:** ICF-only (`use_multi_task_model=False`)

**Recommendations:**
- **Enable language + era tasks**: Stronger signals from auxiliary tasks
- **Use AMOO**: Adaptive multi-objective optimization for task weighting
- **Rationale**: Multi-task learning can improve generalization

### 6. Data & Augmentation

**Current:** `augment_prob=0.2`

**Recommendations:**
- **Increase to 0.3-0.4**: Better generalization
- **Curriculum learning**: Already available, ensure it's enabled
- **Stratified sampling**: Already enabled, good

### 7. Architecture Considerations

**Current:** UniversalICF (emb_dim=16, hidden_dim=128)

**Recommendations:**
- **If underfitting**: Increase `hidden_dim` to 256
- **If overfitting**: Reduce model capacity or increase regularization
- **Consider ResidualICF**: For deeper learning if needed

## Research Integration

### Already Integrated:
- ✅ Soft ranking with `rank-relax`
- ✅ Unified loss framework
- ✅ Multi-task architecture
- ✅ AMOO (Aligned Multi-Objective Optimization)
- ✅ Component-specific learning rates
- ✅ Gradient monitoring

### To Explore:
- Listwise losses (NeuralNDCG, already available)
- Advanced learning rate finders
- Gradient flow analysis
- Model compression (quantization/pruning)

## Immediate Next Steps

1. **Verify early stopping configuration** - Ensure it's monitoring `val_spearman_corr` with appropriate patience
2. **Increase `spearman_weight`** - Try 20.0-30.0 in next experiment
3. **Enable ReduceLROnPlateau** - Monitor `val_spearman_corr` for plateau
4. **Launch multi-task experiment** - Enable language + era tasks with AMOO
5. **Monitor overfitting** - Track train-val gap and adjust dropout if needed

## Experiment Configurations to Try

### Config 1: Higher Spearman Weight
```python
{
    'name': 'multitask_icf_only_high_spearman',
    'spearman_weight': 30.0,
    'spearman_reg_strength': 1.5,
    'use_reduce_lr_on_plateau': True,
    'lr_patience': 5,
    'lr_factor': 0.5,
}
```

### Config 2: Multi-Task with AMOO
```python
{
    'name': 'multitask_icf_lang_era_amoo',
    'use_multi_task_model': True,
    'use_unified_loss': True,
    'use_amoo': True,
    'spearman_weight': 20.0,
}
```

### Config 3: Enhanced Regularization
```python
{
    'name': 'multitask_icf_only_strong_reg',
    'dropout': 0.5,
    'weight_decay': 2e-4,
    'clip_grad_norm': 1.0,
    'spearman_weight': 25.0,
}
```

## Monitoring Recommendations

- Track train-val gap continuously
- Monitor gradient norms for vanishing/exploding gradients
- Use trainctl's ETA and progress tracking
- Compare experiments using trainctl's comparison utilities
- Check storage usage periodically

