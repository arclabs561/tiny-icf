# Parallel Experiments Summary

Generated: 2025-12-05 19:12:18

## Active Experiments

### 1. multitask_icf_only (Baseline)
- **Status**: Running (Epoch 21/150)
- **Best Spearman**: 0.1706 (epoch 8)
- **Current Spearman**: 0.1613
- **Configuration**:
  - spearman_weight: 10.0
  - scheduler_type: cosine (default)
  - dropout: 0.4
  - weight_decay: 1e-4
- **Purpose**: Baseline with unified loss framework

### 2. multitask_icf_high_spearman_plateau
- **Status**: Starting up
- **Configuration**:
  - spearman_weight: 25.0 (2.5x increase)
  - scheduler_type: plateau (adaptive LR)
  - dropout: 0.4
  - weight_decay: 1e-4
- **Purpose**: Test higher ranking emphasis with adaptive scheduling

### 3. multitask_icf_strong_reg
- **Status**: Starting up
- **Configuration**:
  - spearman_weight: 25.0
  - scheduler_type: plateau
  - dropout: 0.5 (increased)
  - weight_decay: 2e-4 (increased)
  - clip_grad_norm: 1.0
- **Purpose**: Test enhanced regularization to reduce overfitting

### 4. multitask_icf_lang_era_amoo
- **Status**: Starting up
- **Configuration**:
  - use_multi_task_model: True
  - output_tasks: ['icf', 'language', 'era']
  - use_amoo: True (adaptive multi-objective)
  - spearman_weight: 20.0
  - scheduler_type: plateau
- **Purpose**: Test multi-task learning with adaptive task weighting

## Expected Improvements

1. **Higher Spearman Weight (25.0)**: Should improve ranking performance
2. **Plateau Scheduler**: Adaptive LR reduction should reduce overfitting
3. **Enhanced Regularization**: Should prevent performance decline after peak
4. **Multi-Task Learning**: Auxiliary tasks should improve generalization

## Monitoring

- Individual logs: `models/<experiment_name>/training.log`
- Unified monitor: `uv run python scripts/monitor_all_experiments.py`
- Status checks: Run status check script anytime

## Next Steps

1. Wait for all experiments to reach 5+ epochs
2. Compare initial performance trends
3. Analyze which configuration performs best
4. Iterate based on results
