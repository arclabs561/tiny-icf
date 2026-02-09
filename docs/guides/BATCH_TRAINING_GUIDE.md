# Non-Interactive Batch Training Guide

## Overview

This guide covers non-interactive batch training using PyTorch Lightning on RunPod, optimized for automated workflows.

## Architecture

### PyTorch Lightning Benefits

- ✅ **Automatic checkpointing**: Resume from interruptions
- ✅ **Mixed precision**: `precision="16-mixed"` flag (vs manual AMP)
- ✅ **Multi-GPU**: `devices=N` one-liner (vs manual DDP)
- ✅ **Minimal boilerplate**: Trainer handles loops, validation, logging
- ✅ **Performance**: ~0.06s/epoch overhead (negligible)

### Non-Interactive Workflow

1. **Create pod** → Upload code → **Run training** → Download results → **Stop pod**
2. Training script runs autonomously and exits cleanly
3. Checkpoints saved to persistent volume or downloaded after completion

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
# Complete workflow: create → train → download → stop
python3 scripts/runpod_batch_training.py --gpu-type "NVIDIA GeForce RTX 4080 SUPER"
```

### Option 2: Shell Script

```bash
./scripts/runpod_batch_complete.sh "NVIDIA GeForce RTX 4080 SUPER"
```

### Option 3: Manual Steps

```bash
# 1. Create pod
POD_ID=$(runpodctl create pod \
    --name tiny-icf-batch \
    --imageName runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 \
    --gpuType "NVIDIA GeForce RTX 4080 SUPER" \
    --containerDiskSize 30)

# 2. Upload project
runpodctl send . $POD_ID /workspace/tiny-icf

# 3. Run training (non-interactive)
runpodctl exec $POD_ID -- bash -c "
cd /workspace/tiny-icf
uv pip install -e .
python scripts/train_batch.py --epochs 50 --batch-size 256
"

# 4. Download results
runpodctl receive $POD_ID /workspace/tiny-icf/models ./models

# 5. Stop pod
runpodctl stop pod $POD_ID
```

## Training Scripts

### `train_lightning.py` - Lightning Training

Full-featured Lightning training with:
- Automatic checkpointing
- Mixed precision (16-mixed)
- Learning rate scheduling
- Early stopping
- Validation metrics (MAE, RMSE, Spearman correlation)
- CSV logging

### `train_batch.py` - Batch Entrypoint

Non-interactive entrypoint designed for container execution:
- Runs training and exits cleanly
- Saves models to `/workspace/models`
- Logs to `training.log`
- Exit code 0 on success

## Configuration

### GPU Selection

Best options for cost/performance:
- **RTX 4080 SUPER**: $0.170/hr spot, 41GB, best value
- **RTX 4090**: $0.200/hr spot, 36GB, fastest
- **RTX 3090**: $0.110/hr spot, 27GB, cheapest

### Training Parameters

Default (optimized for GPU):
- Batch size: 256 (vs 128 vanilla)
- Learning rate: 2e-3 (vs 1e-3)
- Precision: 16-mixed (automatic AMP)
- Data workers: 4 (parallel loading)
- Early stopping: 10 epochs patience

### Persistent Storage

For checkpoints that survive pod termination:
- Use RunPod volumes: `--volume-path /workspace`
- Or S3-compatible storage via RunPod API
- Checkpoints auto-saved every epoch

## Monitoring

### During Training

```bash
# Get SSH command
runpodctl ssh connect <pod-id>

# Then in SSH session:
tail -f /workspace/tiny-icf/training.log
tail -f /workspace/tiny-icf/models/logs/version_0/metrics.csv
```

### Check Training Status

```bash
# Check if training process is running
runpodctl exec <pod-id> -- ps aux | grep train_batch

# Check model files
runpodctl exec <pod-id> -- ls -lh /workspace/models
```

## Best Practices

### 1. Use Persistent Volumes

```bash
# Create pod with volume
runpodctl create pod \
    --volume-path /workspace \
    --gpuType "NVIDIA GeForce RTX 4080 SUPER" \
    ...
```

### 2. Checkpoint Frequently

Lightning auto-saves:
- Best model (lowest val_loss)
- Last checkpoint
- Top-3 checkpoints

### 3. Resume from Checkpoint

If pod is interrupted:
```python
trainer = Trainer(
    ckpt_path="/workspace/models/tiny-icf-epoch=42-val_loss=0.0123.ckpt"
)
trainer.fit(model, datamodule)
```

### 4. Exit Cleanly

Training script should:
- Save final model
- Exit with code 0 (success) or non-zero (failure)
- Log completion status

### 5. Cost Optimization

- Use spot instances: `--bid=0.3`
- Stop pod immediately after training
- Download results before stopping
- Use early stopping to avoid unnecessary epochs

## Comparison: Lightning vs Vanilla

| Feature | Lightning | Vanilla |
|---------|-----------|---------|
| Checkpointing | Automatic | Manual |
| Mixed precision | `precision="16-mixed"` | Manual AMP |
| Multi-GPU | `devices=N` | Manual DDP |
| LR scheduling | Built-in | Manual |
| Validation metrics | Automatic | Manual |
| Code lines | ~150 | ~300+ |
| Overhead | ~0.06s/epoch | Baseline |

## Troubleshooting

### Training Not Starting

```bash
# Check pod status
runpodctl get pod

# Check logs
runpodctl exec <pod-id> -- cat /workspace/tiny-icf/install.log
```

### Out of Memory

- Reduce batch size: `--batch-size 128`
- Use gradient accumulation (Lightning: `accumulate_grad_batches=2`)

### Slow Training

- Increase batch size (if memory allows)
- Use more data workers: `--num-workers 8`
- Enable `torch.backends.cudnn.benchmark = True` (disable deterministic mode)

## Next Steps

1. **Test locally**: `python scripts/train_batch.py` (without RunPod)
2. **Run on pod**: Use `runpod_batch_training.py`
3. **Monitor**: SSH into pod and tail logs
4. **Download**: Get models when complete
5. **Validate**: Use `scripts/validate_trained_model.py`

