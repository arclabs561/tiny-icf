# trainctl Integration Complete

## Summary

We've integrated trainctl utilities into the training pipeline and made Aim required for systematic experiment tracking.

## Changes Made

### 1. Made Aim Required

**Before**: Aim was optional, experiments could run without tracking
**After**: Aim is required - training fails if Aim is not available

**Changes**:
- Updated `train_flexible_opportunistic.py` to require Aim
- Added `aim_experiment` field to all experiment configs
- Updated `pyproject.toml` to include `aim>=3.0.0` as a dependency

**Impact**: All experiments are now systematically tracked in Aim, enabling comparison and analysis.

### 2. Integrated trainctl Utilities

**Added**:
- Checkpoint pruning after training (keeps top-3 + last checkpoint)
- Experiment status tracking via trainctl's `metrics_loader`
- Storage management utilities available for future use

**Changes in `train_flexible_opportunistic.py`**:
```python
# Post-training: Use trainctl utilities for checkpoint pruning
if HAS_TRAINCTL_UTILS:
    removed = cleanup_checkpoints(model_dir, keep_top_k=3, keep_last=True)
    if removed:
        print(f"✅ Pruned {len(removed)} old checkpoints")
```

### 3. Created Experiment Management Scripts

**New Scripts**:
1. **`scripts/compare_experiments.py`**: Compare experiments via Aim or trainctl
   ```bash
   uv run scripts/compare_experiments.py --method both
   uv run scripts/compare_experiments.py --method aim --experiment icf-training
   uv run scripts/compare_experiments.py --method trainctl --storage
   ```

2. **`scripts/archive_completed_experiments.py`**: Archive completed experiments
   ```bash
   uv run scripts/archive_completed_experiments.py --days-old 7
   uv run scripts/archive_completed_experiments.py --experiments exp1 exp2
   uv run scripts/archive_completed_experiments.py --dry-run  # Preview
   ```

3. **`scripts/monitor_with_trainctl.py`**: Unified monitoring using trainctl
   ```bash
   uv run scripts/monitor_with_trainctl.py --follow --interval 10
   uv run scripts/monitor_with_trainctl.py --experiments exp1 exp2
   ```

### 4. Experiment Config Updates

**Added `aim_experiment` field to all configs**:
- Standard experiments: `'aim_experiment': 'icf-training'`
- Distillation experiments: `'aim_experiment': 'icf-distillation'`

This groups related experiments in Aim for easier comparison.

## Usage

### Starting Aim UI

```bash
# Start Aim UI to view experiments
aim up

# Access at http://127.0.0.1:43800
```

### Comparing Experiments

```bash
# Compare all experiments (both Aim and trainctl)
uv run scripts/compare_experiments.py --method both

# Compare only via Aim
uv run scripts/compare_experiments.py --method aim --experiment icf-training

# Compare with storage statistics
uv run scripts/compare_experiments.py --method trainctl --storage
```

### Monitoring Experiments

```bash
# Monitor all experiments (continuous updates)
uv run scripts/monitor_with_trainctl.py --follow

# Monitor specific experiments
uv run scripts/monitor_with_trainctl.py --experiments multitask_icf_only multitask_icf_high_spearman_plateau
```

### Archiving Experiments

```bash
# Archive experiments older than 7 days
uv run scripts/archive_completed_experiments.py --days-old 7

# Archive specific experiments
uv run scripts/archive_completed_experiments.py --experiments exp1 exp2

# Preview what would be archived (dry run)
uv run scripts/archive_completed_experiments.py --days-old 7 --dry-run
```

## Benefits

1. **Systematic Tracking**: All experiments tracked in Aim (required)
2. **Automatic Cleanup**: Checkpoint pruning after training (keeps top-3 + last)
3. **Easy Comparison**: Compare experiments via Aim UI or scripts
4. **Storage Management**: Archive old experiments to save space
5. **Unified Monitoring**: Single interface for all experiments

## Next Steps

1. **Use trainctl CLI for launching**: Launch experiments via `trainctl aws train` instead of direct Python execution
2. **S3 Integration**: Add automatic S3 sync for checkpoints and archives
3. **Experiment Registry**: Create JSON registry of all experiments with metadata
4. **Best Experiment Tracking**: Auto-tag best experiments in Aim

## Status

✅ **Completed**:
- Aim required (not optional)
- trainctl utilities integrated
- Checkpoint pruning after training
- Experiment comparison scripts
- Archive scripts
- Monitoring scripts
- All configs have `aim_experiment` field

⏳ **Next**:
- Launch via `trainctl aws train`
- S3 sync integration
- Experiment registry
- Best experiment auto-tagging

