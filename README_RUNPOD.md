# RunPod Training - Complete Guide

## Overview

This project includes a complete, polished workflow for training the IDF estimation model on RunPod GPUs using PyTorch Lightning and PEP 723 scripts.

## Quick Start

```bash
# One command to start everything
uv run scripts/runpod_batch.py
```

## Architecture

### Core Components

1. **`scripts/train_batch.py`** (PEP 723)
   - Self-contained training script
   - PyTorch Lightning with automatic fallback
   - Mixed precision, curriculum learning, early stopping
   - Run with: `uv run scripts/train_batch.py`

2. **`scripts/runpod_batch.py`** (PEP 723)
   - Pod management (create/find)
   - Project upload
   - Training orchestration
   - Result download
   - Run with: `uv run scripts/runpod_batch.py`

3. **`scripts/runpod_utils.py`**
   - Shared utilities
   - API key extraction from MCP config
   - runpodctl configuration
   - Pod ID extraction

### Training Features

- ✅ **PyTorch Lightning** - Automatic checkpointing, mixed precision, multi-GPU
- ✅ **PEP 723** - Self-contained scripts, no setup needed
- ✅ **Curriculum Learning** - Progressive difficulty
- ✅ **Early Stopping** - Prevents overfitting
- ✅ **Mixed Precision** - 16-bit for faster training
- ✅ **Data Augmentation** - Real typos, multilingual, symbols, emojis

## Usage

### Basic Training

```bash
# Start training (creates pod, uploads, trains)
uv run scripts/runpod_batch.py

# Use existing pod
uv run scripts/runpod_batch.py --pod-id <pod-id>

# Custom GPU
uv run scripts/runpod_batch.py --gpu-type "NVIDIA GeForce RTX 4090"
```

### Workflow Steps

1. **Upload Only**
   ```bash
   uv run scripts/runpod_batch.py --upload-only --pod-id <pod-id>
   ```

2. **Start Training** (if upload-only was used)
   ```bash
   # SSH to pod
   runpodctl ssh connect <pod-id>
   cd /workspace/tiny-icf
   uv run scripts/train_batch.py
   ```

3. **Monitor**
   ```bash
   # Check status
   uv run scripts/runpod_batch.py --status --pod-id <pod-id>
   
   # Or SSH and watch
   runpodctl ssh connect <pod-id>
   tail -f /workspace/tiny-icf/training.log
   ```

4. **Download Results**
   ```bash
   uv run scripts/runpod_batch.py --download-only --pod-id <pod-id>
   ```

## Configuration

### API Key

Automatically extracted from `~/.cursor/mcp.json`. No manual setup needed.

The `runpod_utils.py` module handles:
- JSON parsing (with fallback to regex)
- Nested structure traversal
- Malformed JSON handling

### Training Parameters

Default (optimized for GPU):
- Epochs: 50
- Batch Size: 256
- Learning Rate: 2e-3
- Precision: 16-mixed
- Devices: 1 GPU
- Workers: 4

Override via command-line arguments.

## Output Structure

```
/workspace/tiny-icf/
├── models/
│   ├── tiny-icf-{epoch}-{val_loss}.pt  # Best checkpoints
│   ├── model_final.pt                  # Final model
│   └── logs/                           # Lightning CSV logs
├── training.log                        # Training output
└── training.pid                        # Process ID
```

## Troubleshooting

### Training Won't Start Automatically

The `runpodctl exec` command has timeout issues. Use SSH:

```bash
runpodctl ssh connect <pod-id>
cd /workspace/tiny-icf
uv run scripts/train_batch.py
```

This is the most reliable method.

### Check Pod Status

```bash
runpodctl get pod <pod-id>
```

### View Logs

```bash
runpodctl ssh connect <pod-id>
tail -f /workspace/tiny-icf/training.log
```

### Verify Training Running

```bash
runpodctl ssh connect <pod-id>
ps aux | grep train
cat /workspace/tiny-icf/training.pid
```

## Advanced

### Background Training

```bash
uv run scripts/runpod_batch.py --background --pod-id <pod-id>
```

### Stop Pod After Download

```bash
uv run scripts/runpod_batch.py --download-only --pod-id <pod-id> --stop-after
```

### Use GraphQL API

```bash
uv run scripts/runpod_batch.py --use-api --gpu-type "RTX 4090"
```

## File Organization

### Core Scripts (Keep)
- `scripts/train_batch.py` - Training script
- `scripts/runpod_batch.py` - Orchestration
- `scripts/runpod_utils.py` - Utilities
- `scripts/validate_trained_model.py` - Validation
- `scripts/optimize_training.py` - GPU recommendations

### Experimental Scripts (Can Remove)
Many experimental auto-start scripts were created during development. These can be removed:
- `auto_start_*.py`
- `mcp_*.py` (experimental)
- `quick_*.py`
- `start_*.py` (except if needed)
- `try_*.py`
- `execute_*.py`

## Best Practices

1. **Use PEP 723 scripts** - Self-contained, no setup
2. **SSH for execution** - Most reliable method
3. **Monitor via logs** - `tail -f training.log`
4. **Download regularly** - Don't lose checkpoints
5. **Use background mode** - For long training runs

## Cost Estimation

- **RTX 3090**: $0.22/hr
- **RTX 4080 SUPER**: $0.17/hr (spot)
- **RTX 4090**: $0.20/hr (spot)
- **Training Time**: 2-4 hours (50 epochs)
- **Estimated Cost**: $0.34-$0.88

## Support

For issues:
1. Check pod status: `runpodctl get pod <pod-id>`
2. View logs: SSH and `tail -f training.log`
3. Check training PID: `cat training.pid`
4. Use manual SSH method if automated fails

