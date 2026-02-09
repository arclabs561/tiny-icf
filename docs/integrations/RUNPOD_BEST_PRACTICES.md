# RunPod Non-Interactive Training: Best Practices

## Core Principle: Use `runpodctl exec` for Non-Interactive Jobs

For batch training, **always use `runpodctl exec`** instead of SSH. It's designed for non-interactive execution and returns proper exit codes.

```bash
# ✅ CORRECT: Non-interactive execution
runpodctl exec <pod-id> -- bash -c "
cd /workspace/tiny-icf
uv run scripts/train_batch.py --epochs 50 --batch-size 256
"

# ❌ AVOID: SSH requires interactive session
runpodctl ssh connect <pod-id>  # Then manually run commands
```

## PyTorch Lightning vs Vanilla PyTorch

**Use PyTorch Lightning for non-interactive training.** The benefits far outweigh the minimal overhead:

| Feature | Lightning | Vanilla | Impact |
|---------|-----------|---------|--------|
| **Checkpointing** | Automatic (`ModelCheckpoint`) | Manual save/load logic | Critical for cloud interruptions |
| **Mixed precision** | `precision="16-mixed"` flag | Manual `torch.cuda.amp` context | 2x speedup, minimal code |
| **Multi-GPU** | `devices="auto"` | Manual DDP/NCCL setup | Essential for scaling |
| **Resume training** | `trainer.fit(ckpt_path="...")` | Manual state restoration | Saves hours of lost work |
| **Overhead** | ~0.06s/epoch | Baseline | Negligible |

### When to Use Vanilla PyTorch

Only if:
- Lightning adds unacceptable complexity for your specific use case
- You need fine-grained control over training loop internals
- You're debugging low-level CUDA operations

For 99% of training jobs, Lightning is the better choice.

## Non-Interactive Workflow Pattern

### 1. Create Pod with Persistent Volume

```bash
# Create pod with volume for checkpoints
runpodctl create pod \
    --name tiny-icf-batch-$(date +%s) \
    --imageName runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 \
    --gpuType "NVIDIA GeForce RTX 4080 SUPER" \
    --containerDiskSize 30 \
    --volumePath /workspace \
    --env "PYTHONUNBUFFERED=1"
```

**Key points:**
- `--volumePath /workspace`: Checkpoints survive pod termination
- `PYTHONUNBUFFERED=1`: Real-time log streaming
- Use spot instances with `--bid=0.3` for cost savings

### 2. Upload Code (Smart Exclusions)

```bash
# Upload only essential files (excludes .venv, __pycache__, .git)
runpodctl send <pod-id> . /workspace/tiny-icf
```

Or use your existing `upload_project()` function which excludes unnecessary files.

### 3. Execute Training Non-Interactively

```bash
# Single command that runs and exits
runpodctl exec <pod-id> -- bash -c "
cd /workspace/tiny-icf
uv run scripts/train_batch.py \
    --data data/multilingual/multilingual_combined.csv \
    --typo-corpus data/typos/github_typos.csv \
    --output-dir /workspace/models \
    --epochs 50 \
    --batch-size 256 \
    --devices 1 \
    --precision 16-mixed
"
```

**Critical:** The command should:
- Exit with code 0 on success, non-zero on failure
- Save checkpoints to persistent volume (`/workspace/models`)
- Log to file for monitoring: `2>&1 | tee training.log`

### 4. Monitor Training (Optional)

```bash
# Check if training is running
runpodctl exec <pod-id> -- ps aux | grep train_batch

# View logs
runpodctl exec <pod-id> -- tail -n 100 /workspace/tiny-icf/training.log

# Check checkpoint directory
runpodctl exec <pod-id> -- ls -lh /workspace/models
```

### 5. Download Results

```bash
# Download models when training completes
runpodctl receive <pod-id> /workspace/models ./models

# Stop pod to avoid charges
runpodctl stop pod <pod-id>
```

## Improved Training Script Pattern

Your `train_batch.py` is already good, but ensure it:

1. **Exits cleanly** with proper exit codes:
```python
def main():
    try:
        trainer.fit(model, datamodule)
        print("✅ Training complete")
        return 0
    except KeyboardInterrupt:
        print("⚠️  Training interrupted")
        return 130
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
```

2. **Saves to persistent volume**:
```python
# Use /workspace/models (on volume) not /tmp
output_dir = Path("/workspace/models")
checkpoint_callback = ModelCheckpoint(
    dirpath=output_dir,
    filename="tiny-icf-{epoch:02d}-{val_loss:.4f}",
    save_top_k=3,
    save_last=True,
)
```

3. **Handles resume from checkpoint**:
```python
# If checkpoint exists, resume automatically
ckpt_path = args.output_dir / "last.ckpt"
if ckpt_path.exists():
    trainer.fit(model, datamodule, ckpt_path=str(ckpt_path))
else:
    trainer.fit(model, datamodule)
```

## Cost Optimization

### Use Spot Instances

```bash
# Create spot pod (up to 70% cheaper)
runpodctl create pod \
    --gpuType "NVIDIA GeForce RTX 4080 SUPER" \
    --bid 0.3 \
    ...
```

**Trade-off:** Pod can be preempted. Lightning's auto-checkpointing handles this.

### Stop Pod Immediately After Training

```bash
# In your orchestration script
runpodctl exec <pod-id> -- bash -c "..." && \
runpodctl receive <pod-id> /workspace/models ./models && \
runpodctl stop pod <pod-id>
```

### Use Early Stopping

Lightning's `EarlyStopping` callback prevents unnecessary epochs:
```python
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
)
```

## Error Handling Best Practices

### 1. Check Pod Status Before Execution

```python
def check_pod_ready(pod_id: str) -> bool:
    """Verify pod is RUNNING before executing commands."""
    result = subprocess.run(
        ["runpodctl", "get", "pod", pod_id],
        capture_output=True,
        text=True,
    )
    return "RUNNING" in result.stdout
```

### 2. Handle Preemption Gracefully

```python
# In training script
try:
    trainer.fit(model, datamodule)
except RuntimeError as e:
    if "CUDA" in str(e) or "preempted" in str(e).lower():
        print("⚠️  Pod preempted, checkpoints saved")
        # Exit gracefully - can resume later
        sys.exit(0)
    raise
```

### 3. Validate Training Completion

```python
def validate_training_complete(pod_id: str) -> bool:
    """Check if training completed successfully."""
    result = subprocess.run(
        ["runpodctl", "exec", pod_id, "--", "test", "-f", "/workspace/models/model_final.pt"],
        capture_output=True,
    )
    return result.returncode == 0
```

## Complete Orchestration Script Pattern

Here's the ideal pattern for your `runpod_batch.py`:

```python
def run_training_non_interactive(pod_id: str) -> bool:
    """Execute training non-interactively using runpodctl exec."""
    train_cmd = """
cd /workspace/tiny-icf
uv run scripts/train_batch.py \
    --data data/multilingual/multilingual_combined.csv \
    --typo-corpus data/typos/github_typos.csv \
    --output-dir /workspace/models \
    --epochs 50 \
    --batch-size 256 \
    --devices 1 \
    --precision 16-mixed \
    2>&1 | tee /workspace/tiny-icf/training.log
"""
    
    try:
        result = subprocess.run(
            ["runpodctl", "exec", pod_id, "--", "bash", "-c", train_cmd],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        
        if result.returncode == 0:
            print("✅ Training completed successfully")
            return True
        else:
            print(f"❌ Training failed (exit code {result.returncode})")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Training timeout (still running in background)")
        print("   Check logs: runpodctl exec <pod-id> -- tail -f /workspace/tiny-icf/training.log")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Execution failed: {e}")
        return False
```

## Monitoring Without SSH

You don't need SSH for monitoring:

```python
def get_training_status(pod_id: str) -> dict:
    """Get training status without SSH."""
    status = {}
    
    # Check if process is running
    result = subprocess.run(
        ["runpodctl", "exec", pod_id, "--", "pgrep", "-f", "train_batch"],
        capture_output=True,
    )
    status["running"] = result.returncode == 0
    
    # Get last 10 lines of log
    result = subprocess.run(
        ["runpodctl", "exec", pod_id, "--", "tail", "-n", "10", "/workspace/tiny-icf/training.log"],
        capture_output=True,
        text=True,
    )
    status["log_tail"] = result.stdout
    
    # Check checkpoint directory
    result = subprocess.run(
        ["runpodctl", "exec", pod_id, "--", "ls", "-1", "/workspace/models"],
        capture_output=True,
        text=True,
    )
    status["checkpoints"] = result.stdout.strip().split("\n") if result.returncode == 0 else []
    
    return status
```

## Docker Container Approach (Alternative)

For more control, use a Docker container with entrypoint:

```dockerfile
# Dockerfile.batch
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN pip install uv

WORKDIR /workspace/tiny-icf
COPY . .

RUN uv pip install --system -e .

# Entrypoint runs training and exits
ENTRYPOINT ["python", "scripts/train_batch.py"]
CMD ["--epochs", "50", "--batch-size", "256"]
```

Then create pod with custom image:
```bash
runpodctl create pod \
    --imageName your-registry/tiny-icf-training:latest \
    --gpuType "NVIDIA GeForce RTX 4080 SUPER" \
    ...
```

**Advantage:** Training starts automatically when pod is created.

## Summary: Key Takeaways

1. **Use `runpodctl exec`** for all non-interactive commands
2. **Use PyTorch Lightning** for automatic checkpointing and multi-GPU
3. **Save to persistent volumes** (`/workspace/models`) not container disk
4. **Exit codes matter** - check return codes in orchestration scripts
5. **Use spot instances** with `--bid` for cost savings
6. **Stop pods immediately** after training completes
7. **Monitor via `exec`** not SSH - it's non-interactive and scriptable

## Your Current Implementation

Your existing code is already following most best practices:
- ✅ Using Lightning (`train_lightning.py`, `train_batch.py`)
- ✅ PEP 723 scripts for portability
- ✅ Non-interactive execution pattern
- ✅ Checkpoint saving

**Improvements to consider:**
- Use `runpodctl exec` consistently (some scripts still use SSH)
- Add explicit exit code checking
- Add resume-from-checkpoint logic
- Use persistent volumes more consistently

