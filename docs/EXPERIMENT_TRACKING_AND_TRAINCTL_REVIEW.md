# Experiment Tracking and trainctl Usage Review

## Current State

### Experiment Tracking

**What We Have:**
1. **CSVLogger** (always enabled): Logs metrics to CSV files in `models/<experiment>/lightning_logs/version_0/metrics.csv`
2. **TensorBoardLogger** (optional): If TensorBoard is available, logs to TensorBoard format
3. **AimLogger** (optional): If Aim is available, logs to Aim repository (`~/.aim` by default)
4. **Custom monitoring scripts**: `scripts/monitor_all_experiments.py`, `scripts/monitor_training.sh`

**What Gets Tracked:**
- Training metrics: `train_loss`, `train_huber_loss`, `train_ranking_loss`, `train_spearman_ratio`
- Validation metrics: `val_loss`, `val_spearman_corr`, `val_mae`, `val_rmse`
- Learning rate: `learning_rate` (when using schedulers)
- Gradient norms: `grad_norm`, `grad_norm_min`, `grad_norm_max`, per-layer gradients
- Loss components: Individual task losses, distillation losses, etc.
- Hyperparameters: All config parameters (via PyTorch Lightning's `log_hyperparams`)

**Current Implementation:**
```python
# In train_flexible_opportunistic.py
loggers = [CSVLogger(model_dir, name='lightning_logs')]  # CSV always needed

if HAS_TENSORBOARD:
    loggers.append(TensorBoardLogger(model_dir, name='lightning_logs', version=0))

if HAS_AIM and config.get('use_aim', False):
    aim_logger = AimLogger(
        experiment=config.get('aim_experiment', 'icf-training'),
        run_hash=experiment_name,
    )
    loggers.append(aim_logger)
```

### trainctl Usage

**What trainctl Provides:**
1. **Resource Management**: AWS EC2, RunPod, local training orchestration
2. **Checkpoint Management**: List, inspect, resume from checkpoints
3. **Storage Management**: S3 operations, EBS volume management
4. **Monitoring**: Real-time monitoring with `trainctl monitor`, `trainctl top`
5. **Cost Optimization**: Spot instances, resource cleanup

**Current Usage:**
- ✅ Training script is in `../trainctl/training/scripts/train_flexible_opportunistic.py`
- ✅ Uses trainctl's directory structure (`models/<experiment>/`)
- ❌ **NOT using trainctl's checkpoint management** (using PyTorch Lightning's ModelCheckpoint instead)
- ❌ **NOT using trainctl's storage utilities** (no automatic S3 sync, no checkpoint pruning)
- ❌ **NOT using trainctl's monitoring** (using custom Python scripts instead)
- ❌ **NOT using trainctl's resource management** (not launching via `trainctl aws train`)

## Issues and Gaps

### 1. Experiment Tracking Fragmentation

**Problem**: Metrics are scattered across:
- CSV files in `models/<experiment>/lightning_logs/version_0/metrics.csv`
- TensorBoard logs (if enabled)
- Aim repository (if enabled)
- Custom monitoring scripts reading CSV files

**Impact**: 
- No single source of truth for experiment comparison
- Difficult to systematically compare experiments
- Manual aggregation required for analysis

### 2. trainctl Underutilization

**Problem**: We're not using trainctl's core features:
- Checkpoint management: Using PyTorch Lightning's ModelCheckpoint instead of trainctl's checkpoint system
- Storage management: No automatic S3 sync, no checkpoint pruning
- Resource management: Not launching via `trainctl aws train`
- Monitoring: Using custom Python scripts instead of `trainctl monitor`

**Impact**:
- Missing trainctl's checkpoint pruning (keeps last N, archives old)
- Missing trainctl's storage optimization (S3 sync, EBS management)
- Missing trainctl's resource tracking (cost, utilization)
- Missing trainctl's unified monitoring dashboard

### 3. No Systematic Experiment Comparison

**Problem**: No automated way to:
- Compare experiments side-by-side
- Track best experiments over time
- Identify which hyperparameters work best
- Archive completed experiments

**Impact**:
- Manual comparison required
- No systematic tracking of what works
- Risk of losing track of good experiments

## Recommendations

### Priority 1: Integrate trainctl Properly

1. **Use trainctl's checkpoint system**:
   ```python
   from trainctl.utils.checkpoint_manager import CheckpointManager
   
   checkpoint_manager = CheckpointManager(
       checkpoint_dir=model_dir / "checkpoints",
       save_interval=5,  # Save every 5 epochs
       keep_last_n=10,  # Keep last 10 checkpoints
   )
   ```

2. **Use trainctl's storage utilities**:
   ```python
   from trainctl.utils.storage_manager import StorageManager
   
   storage = StorageManager(
       s3_bucket=config.get('s3_bucket'),
       experiment_name=experiment_name,
   )
   
   # Auto-sync checkpoints to S3
   storage.sync_checkpoints_to_s3(model_dir / "checkpoints")
   ```

3. **Launch via trainctl**:
   ```bash
   # Instead of running Python script directly
   trainctl aws train $INSTANCE_ID \
       ../trainctl/training/scripts/train_flexible_opportunistic.py \
       --experiments multitask_icf_only \
       --sync-code
   ```

4. **Use trainctl's monitoring**:
   ```bash
   # Instead of custom Python scripts
   trainctl monitor --log models/*/training.log --follow
   trainctl top --interval 5  # Interactive dashboard
   ```

### Priority 2: Unify Experiment Tracking

1. **Standardize on Aim as primary tracker**:
   - Always enable Aim (make it required, not optional)
   - Use Aim for all experiment comparison
   - Keep CSV as backup for custom scripts

2. **Create experiment comparison script**:
   ```python
   # scripts/compare_experiments.py
   from aim import Run
   
   runs = Run.filter(experiment="icf-training")
   # Compare metrics, hyperparameters, etc.
   ```

3. **Automate experiment archival**:
   - Archive completed experiments to S3
   - Keep only active experiments locally
   - Use trainctl's storage manager for this

### Priority 3: Systematic Experiment Management

1. **Experiment registry**:
   - Maintain a JSON/YAML file of all experiments
   - Track: name, config, status, best metrics, location
   - Update automatically when experiments complete

2. **Best experiment tracking**:
   - Automatically identify best experiments by metric
   - Tag best experiments in Aim
   - Archive best models separately

3. **Hyperparameter search integration**:
   - Use Aim's hyperparameter search features
   - Or integrate with Optuna/Hyperopt via Aim

## Implementation Plan

### Phase 1: trainctl Integration (1-2 days)

1. **Update training script to use trainctl utilities**:
   - Import trainctl's checkpoint manager
   - Import trainctl's storage manager
   - Replace PyTorch Lightning ModelCheckpoint with trainctl's system
   - Add S3 sync for checkpoints

2. **Update launch scripts**:
   - Use `trainctl aws train` instead of direct Python execution
   - Use `trainctl monitor` instead of custom scripts

3. **Test trainctl integration**:
   - Verify checkpoint management works
   - Verify S3 sync works
   - Verify monitoring works

### Phase 2: Unified Tracking (2-3 days)

1. **Make Aim required**:
   - Remove optional Aim flag
   - Always initialize AimLogger
   - Ensure all metrics are logged to Aim

2. **Create experiment comparison tools**:
   - Script to compare experiments in Aim
   - Script to identify best experiments
   - Script to archive completed experiments

3. **Update monitoring scripts**:
   - Use Aim API instead of CSV parsing
   - Create unified dashboard using Aim UI

### Phase 3: Systematic Management (1-2 days)

1. **Experiment registry**:
   - Create JSON registry of all experiments
   - Auto-update on experiment start/complete
   - Query registry for experiment status

2. **Best experiment tracking**:
   - Auto-tag best experiments in Aim
   - Archive best models to S3
   - Maintain best experiment list

3. **Documentation**:
   - Update README with trainctl usage
   - Document experiment tracking workflow
   - Document how to compare experiments

## Current Status Summary

| Feature | Status | Notes |
|---------|--------|-------|
| CSV Logger | ✅ Working | Always enabled, used by monitoring scripts |
| TensorBoard | ⚠️ Optional | Available but not always enabled |
| Aim | ⚠️ Optional | Available but not always enabled |
| trainctl Checkpoints | ❌ Not Used | Using PyTorch Lightning's ModelCheckpoint |
| trainctl Storage | ❌ Not Used | No S3 sync, no checkpoint pruning |
| trainctl Monitoring | ❌ Not Used | Using custom Python scripts |
| trainctl Resources | ❌ Not Used | Not launching via `trainctl aws train` |
| Experiment Comparison | ❌ Manual | No automated comparison tools |
| Experiment Registry | ❌ Missing | No systematic tracking |

## Next Steps

1. **Immediate**: Review trainctl utilities and integrate checkpoint/storage managers
2. **Short-term**: Make Aim required, create comparison tools
3. **Long-term**: Build experiment registry, automate archival

