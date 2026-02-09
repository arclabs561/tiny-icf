# RunPod Training Guide

## Quick Start

### 1. Start Training (One Command)

```bash
uv run scripts/runpod_batch.py
```

This will:
- ✅ Configure runpodctl from MCP config
- ✅ Find or create a pod
- ✅ Upload project files
- ✅ Start training in background
- ✅ Show monitoring instructions

### 2. Monitor Training

```bash
# Check status
uv run scripts/runpod_batch.py --status --pod-id <pod-id>

# Or SSH and watch logs
runpodctl ssh connect <pod-id>
tail -f /workspace/tiny-icf/training.log
```

### 3. Download Results

```bash
uv run scripts/runpod_batch.py --download-only --pod-id <pod-id>
```

## Manual Start (If Needed)

If automated start fails, use SSH:

```bash
runpodctl ssh connect <pod-id>
cd /workspace/tiny-icf
uv run scripts/train_batch.py --epochs 50 --batch-size 256
```

## Scripts

### Core Scripts

- **`scripts/train_batch.py`** - PEP 723 training script (Lightning + fallback)
- **`scripts/runpod_batch.py`** - Orchestration (pod management, upload, training)
- **`scripts/runpod_utils.py`** - Shared utilities (API key, config)

### Training Script

The training script (`train_batch.py`) is PEP 723 compliant:
- Self-contained dependencies
- Uses `uv run` for execution
- PyTorch Lightning with automatic fallback
- Mixed precision (16-mixed)
- Curriculum learning
- Early stopping
- Checkpointing

## Configuration

### API Key

Automatically extracted from `~/.cursor/mcp.json`. No manual configuration needed.

### Training Parameters

Default settings (optimized for GPU):
- **Epochs**: 50
- **Batch Size**: 256
- **Learning Rate**: 2e-3
- **Precision**: 16-mixed
- **Devices**: 1 GPU
- **Workers**: 4

Override with command-line arguments.

## Output

Training outputs:
- **Models**: `/workspace/tiny-icf/models/`
  - `tiny-icf-{epoch}-{val_loss}.pt` - Best checkpoints
  - `model_final.pt` - Final model
- **Logs**: `/workspace/tiny-icf/training.log`
- **CSV Logs**: `/workspace/tiny-icf/models/logs/` (Lightning metrics)

## Troubleshooting

### Training Won't Start Automatically

Use manual SSH method (see above). This is the most reliable approach.

### Check Pod Status

```bash
runpodctl get pod <pod-id>
```

### View Training Logs

```bash
runpodctl ssh connect <pod-id>
tail -f /workspace/tiny-icf/training.log
```

### Check if Training is Running

```bash
runpodctl ssh connect <pod-id>
ps aux | grep train
cat /workspace/tiny-icf/training.pid
```

## Advanced Usage

### Custom GPU Type

```bash
uv run scripts/runpod_batch.py --gpu-type "NVIDIA GeForce RTX 4090"
```

### Upload Only

```bash
uv run scripts/runpod_batch.py --upload-only --pod-id <pod-id>
```

### Background Training

```bash
uv run scripts/runpod_batch.py --background --pod-id <pod-id>
```

### Stop Pod After Download

```bash
uv run scripts/runpod_batch.py --download-only --pod-id <pod-id> --stop-after
```

## Architecture

```
runpod_batch.py (orchestration)
    ├── runpod_utils.py (shared utilities)
    └── train_batch.py (training script)
            ├── lightning_module.py (Lightning wrapper)
            ├── lightning_data.py (DataModule)
            └── train_lightning.py (Lightning training)
```

All scripts are PEP 723 compliant and use `uv run` for dependency management.
