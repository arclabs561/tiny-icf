# Ephemeral Pod Setup Guide

## The Challenge

RunPod instances are **ephemeral** - they can be destroyed and recreated, losing all state. This means:
- No persistent storage (except what we sync)
- Limited disk space (often 5GB overlay filesystem)
- Need to set up environment from scratch each time
- Dependencies must be installed fresh

## Current Issues

1. **Disk Space**: 5GB overlay filesystem fills up quickly with PyTorch CUDA (~2GB)
2. **Venv Isolation**: `uv venv` sometimes creates venvs without pip
3. **Dependency Conflicts**: Aim has dependency issues on some pods
4. **Cache Bloat**: uv/pip caches can consume significant space

## Solutions

### 1. Use CPU-Only PyTorch (Space-Saving)

For ephemeral pods with limited space, use CPU-only PyTorch:

```bash
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio
```

**Saves**: ~1.5GB vs CUDA version

### 2. Clean Cache Aggressively

```bash
# Clean before setup
rm -rf ~/.cache/uv ~/.cache/pip
rm -rf .venv

# Use --no-cache-dir with pip
pip install --no-cache-dir ...
```

### 3. Use `uv pip` Directly

Instead of activating venv and using pip, use `uv pip` with `--python`:

```bash
uv pip install --python .venv/bin/python torch numpy pandas
```

This avoids venv activation issues.

### 4. Skip Optional Dependencies

For training, we don't always need Aim:

```bash
# Install core package without optional deps
uv pip install --python .venv/bin/python -e . --no-deps
```

### 5. Setup Script for Ephemeral Pods

Use `scripts/runpod_setup_ephemeral.sh` which:
- Cleans disk space
- Installs uv if needed
- Creates fresh venv
- Installs CPU-only PyTorch
- Installs core dependencies
- Verifies installation

## Quick Setup Command

```bash
# On ephemeral pod
cd /root/idf-est
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Clean and setup
rm -rf ~/.cache/uv ~/.cache/pip .venv
uv venv --python python3.12

# Install CPU-only PyTorch (saves space)
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# Install other deps
uv pip install --python .venv/bin/python \
    numpy pandas tqdm scipy

# Install package
uv pip install --python .venv/bin/python -e . --no-deps

# Verify
.venv/bin/python -c "import torch, numpy, pandas, tiny_icf; print('✓ Ready')"
```

## Training on Ephemeral Pod

```bash
# Start training (no venv activation needed)
.venv/bin/python scripts/train_research_loss.py \
    --data data/word_frequency.csv \
    --epochs 100 \
    --batch-size 32 \
    --use-neural-ndcg \
    --output models/model_neural_ndcg_100ep.pt \
    > logs/training.log 2>&1 &
```

## Monitoring

```bash
# Check if running
ps aux | grep train_research_loss

# Check progress
tail -f logs/training.log

# Check disk space
df -h /
```

## Disk Space Management

### Check Usage
```bash
df -h /
du -sh ~/.cache/uv ~/.cache/pip .venv
```

### Clean Up
```bash
# Remove caches
rm -rf ~/.cache/uv ~/.cache/pip

# Remove venv (recreate if needed)
rm -rf .venv

# Remove old models/logs if needed
rm -rf models/*.pt logs/*.log
```

## Best Practices for Ephemeral Pods

1. **Always sync code first**: `just sync`
2. **Clean before setup**: Remove old venv and caches
3. **Use CPU-only PyTorch**: Saves significant space
4. **Skip optional deps**: Only install what's needed
5. **Monitor disk space**: Check before/after setup
6. **Use background execution**: `nohup` for long training
7. **Save models immediately**: Download before pod dies

## Troubleshooting

### "No space left on device"
- Clean caches: `rm -rf ~/.cache/uv ~/.cache/pip`
- Use CPU-only PyTorch
- Remove old venv: `rm -rf .venv`

### "ModuleNotFoundError"
- Use `uv pip install --python .venv/bin/python` instead of venv activation
- Verify with `.venv/bin/python -c "import ..."`

### "uv: command not found"
- Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Add to PATH: `export PATH="$HOME/.local/bin:$PATH"`

### Venv has no pip
- Use `uv pip` directly with `--python .venv/bin/python`
- Don't rely on venv activation

