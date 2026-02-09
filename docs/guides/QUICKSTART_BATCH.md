# Quick Start: Batch Training on RunPod

## PEP 723 Scripts (Simplified)

All training scripts are now PEP 723 compliant - just run with `uv run`!

## One-Command Training

```bash
# Complete workflow: create pod → upload → train → download
uv run scripts/runpod_batch.py --gpu-type "NVIDIA GeForce RTX 4080 SUPER"
```

## Step-by-Step

### 1. Start Training

```bash
uv run scripts/runpod_batch.py
```

This will:
- Auto-configure runpodctl from `~/.cursor/mcp.json`
- Create/find a pod (RTX 4080 SUPER by default)
- Upload project (excludes .venv)
- Start training in background
- Show SSH command for monitoring

### 2. Monitor Training

```bash
# Get SSH command (shown after starting)
runpodctl ssh connect <pod-id>

# Then in SSH session:
tail -f /workspace/tiny-icf/training.log
```

### 3. Download Results

```bash
uv run scripts/runpod_batch.py --download-only --pod-id <pod-id>
```

### 4. Stop Pod

```bash
runpodctl stop pod <pod-id>
```

## Training Script (PEP 723)

The training script (`train_batch.py`) is self-contained:

```bash
# On the pod (after upload):
cd /workspace/tiny-icf
uv run scripts/train_batch.py --epochs 50 --batch-size 256
```

**No installation needed!** `uv run` automatically:
- Creates isolated environment
- Installs dependencies from PEP 723 metadata
- Runs the script

## What Gets Trained

- **Model**: UniversalICF (33k parameters)
- **Data**: 400k words, multilingual + typos
- **Training**: PyTorch Lightning with mixed precision
- **Output**: `/workspace/models/model_final.pt`

## Benefits of PEP 723

✅ **No pyproject.toml needed** - dependencies in script  
✅ **No virtualenv setup** - `uv run` handles it  
✅ **Portable** - works on any system with `uv`  
✅ **Simple** - one command to run  

## Fallback

If Lightning isn't available, the script automatically falls back to optimized vanilla training (`train_optimized.py`).

