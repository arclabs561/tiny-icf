# Training Scripts Inventory

## Main Training Scripts (Active)

### `scripts/train_flexible_opportunistic.py`
**Status**: ✅ Primary training script
**Purpose**: Main training entry point with PyTorch Lightning integration
**Features**:
- Automatic resource detection (CPU/GPU)
- Multiple experiment configurations
- PyTorch Lightning for all training scenarios
- S3 checkpoint syncing
- ResidualICF model support

### `scripts/scale_gpu_training.sh`
**Status**: ✅ Active
**Purpose**: Launch and manage AWS GPU instances for training
**Features**:
- Spot/On-demand instance management
- Automatic AMI selection
- Project upload and training launch
- Cleanup and monitoring

### `scripts/monitor_aws_training.sh`
**Status**: ✅ Active
**Purpose**: Monitor training progress on AWS instances

### `scripts/show_training_results.sh`
**Status**: ✅ Active
**Purpose**: Display training results from AWS instances

### `scripts/run_flexible_training.sh`
**Status**: ✅ Active
**Purpose**: Wrapper script to start flexible training

### `scripts/monitor_residual_experiments.sh`
**Status**: ✅ Active
**Purpose**: Monitor ResidualICF experiments

## Other Training Scripts

These scripts may be outdated or experimental. Review before archiving.

