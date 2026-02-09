# Justfile Usage Guide

## Quick Start

```bash
# List all available commands
just --list

# Test SSH connection (must work first!)
just test-connection

# Deploy code and setup environment
just deploy

# Run ablation study
just ablation

# Train with different methods
just train-diffsort
just train-listwise
just train-best

# Download results
just download

# Watch training logs
just watch train_diffsort.log

# Check status
just status
```

## SSH Setup Required

**Before using any commands, ensure SSH key is added to RunPod:**

1. Get your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

2. Add to RunPod:
   - Go to RunPod dashboard
   - Navigate to SSH Keys section
   - Add your public key

3. Test connection:
   ```bash
   ssh root@205.196.19.18 -p 11859 -i ~/.ssh/id_ed25519
   ```

## Commands

### `just sync`
Syncs code to RunPod using rsync. Excludes:
- `.git/`
- `__pycache__/`
- `*.pyc`
- `models/*.pt`
- `*.log`

### `just setup`
Installs dependencies on RunPod:
- `uv` (if not present)
- `diffsort`, `torch`, `numpy`, `pandas`, `tqdm`, `scipy`, `lightning`

### `just deploy`
Runs `sync` then `setup` in sequence.

### `just ablation [args]`
Runs ablation study comparing all loss methods:
- Huber only
- Pairwise ranking (weight=2.0, 10.0)
- Listwise LambdaRank
- Listwise ApproxNDCG
- Differentiable sorting (diffsort)

Example:
```bash
just ablation --epochs 20 --batch-size 128
```

### `just train-diffsort [args]`
Trains with differentiable sorting loss.

Example:
```bash
just train-diffsort --epochs 100 --batch-size 32
```

### `just train-listwise [args]`
Trains with listwise ranking loss.

### `just train-best [args]`
Trains with all best practices enabled.

### `just download`
Downloads results from RunPod:
- Models: `models/*.pt`
- Results: `*.json`
- Logs: `*.log`

### `just watch <logfile>`
Watches a log file in real-time.

Example:
```bash
just watch ablation_run.log
```

### `just status`
Shows RunPod system status:
- Python version
- UV version
- GPU info
- Project status

## Configuration

Edit the justfile to change:
- `runpod_host`: SSH hostname
- `runpod_port`: SSH port
- `runpod_key`: SSH key path
- `runpod_path`: Remote project path

## Troubleshooting

### Permission Denied
- Verify SSH key is added to RunPod
- Check key file permissions: `chmod 600 ~/.ssh/id_ed25519`
- Test manually: `ssh root@205.196.19.18 -p 11859 -i ~/.ssh/id_ed25519`

### Connection Timeout
- Check RunPod instance is running
- Verify IP and port are correct
- Check firewall settings

### Module Not Found
- Run `just setup` to install dependencies
- Or manually: `just sync && just setup`
