# GPU Scaling Quick Start

## Commands

```bash
# Launch GPU instance (8 hours, with S3 sync)
./scripts/scale_gpu_training.sh up g4dn.xlarge 8 my-bucket

# Upload project
./scripts/scale_gpu_training.sh upload

# Check status
./scripts/scale_gpu_training.sh status

# Cleanup
./scripts/scale_gpu_training.sh down

# Sync checkpoints manually
./scripts/sync_checkpoints_s3.sh upload my-bucket
```

## Features

- **Auto-cleanup**: Instance shuts down after max-hours
- **Spot interruption**: Auto-checkpoint on 2-minute warning
- **S3 sync**: Optional checkpoint persistence
- **Safe cleanup**: Dry-run mode, selective termination

## Files

- `scripts/scale_gpu_training.sh` - Main scaling script (simplified)
- `scripts/cleanup_aws_resources.sh` - Resource cleanup (simplified)
- `scripts/sync_checkpoints_s3.sh` - Checkpoint sync (simplified)

