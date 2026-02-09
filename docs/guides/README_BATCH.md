# Batch Training on RunPod - Simplified

## PEP 723 Scripts

All scripts use PEP 723 inline metadata - no setup needed, just run with `uv run`.

## Quick Start

```bash
# One command: create → upload → train → monitor
uv run scripts/runpod_batch.py
```

## What This Does

1. **Auto-configures** runpodctl from `~/.cursor/mcp.json`
2. **Creates/finds** pod (RTX 4080 SUPER by default)
3. **Uploads** project (smart, excludes .venv)
4. **Starts training** in background (PyTorch Lightning)
5. **Shows** SSH command for monitoring

## Training Script

The training script (`train_batch.py`) is self-contained:

```bash
# On pod (after upload):
cd /workspace/tiny-icf
uv run scripts/train_batch.py
```

**No installation needed!** Dependencies are in the script metadata.

## Features

- ✅ **PEP 723**: Self-contained scripts, no setup
- ✅ **PyTorch Lightning**: Automatic checkpointing, mixed precision
- ✅ **Non-interactive**: Runs and exits cleanly
- ✅ **Auto-fallback**: Uses vanilla training if Lightning unavailable
- ✅ **Smart upload**: Excludes .venv, only essential files

## Monitoring

```bash
# Get SSH command (shown after starting)
runpodctl ssh connect <pod-id>

# Then:
tail -f /workspace/tiny-icf/training.log
```

## Download Results

```bash
uv run scripts/runpod_batch.py --download-only --pod-id <pod-id>
```

## GPU Options

```bash
# RTX 4090 (fastest)
uv run scripts/runpod_batch.py --gpu-type "NVIDIA GeForce RTX 4090"

# RTX 3090 (cheapest)
uv run scripts/runpod_batch.py --gpu-type "NVIDIA GeForce RTX 3090"
```

## Training Output

Models saved to `/workspace/models/`:
- `model_final.pt` - Final model
- `tiny-icf-epoch=XX-val_loss=X.XXXX.ckpt` - Best checkpoints
- `logs/version_0/metrics.csv` - Training metrics

