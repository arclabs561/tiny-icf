# RunPod Non-Interactive Training: Quick Reference

## TL;DR: Best Practices

1. **Use `runpodctl exec`** - Never use SSH for non-interactive jobs
2. **Use PyTorch Lightning** - Automatic checkpointing, mixed precision, multi-GPU
3. **Save to persistent volumes** - Use `/workspace` not container disk
4. **Check exit codes** - Validate training completion
5. **Use spot instances** - Add `--bid 0.3` for 70% cost savings

## One-Command Training

```bash
# Complete workflow: create → upload → train → download → stop
uv run scripts/runpod_batch.py --gpu-type "NVIDIA GeForce RTX 4080 SUPER"
```

## Common Commands

### Start Training (Background)

```bash
uv run scripts/runpod_batch.py --background
```

### Check Status

```bash
uv run scripts/runpod_batch.py --status --pod-id <pod-id>
```

### Download Results

```bash
uv run scripts/runpod_batch.py --download-only --pod-id <pod-id>
```

### Stop Pod

```bash
runpodctl stop pod <pod-id>
```

## Manual Non-Interactive Execution

```bash
# Execute training command directly
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

## Monitoring Without SSH

```bash
# Check if training is running
runpodctl exec <pod-id> -- pgrep -f train_batch

# View logs
runpodctl exec <pod-id> -- tail -n 100 /workspace/tiny-icf/training.log

# Check checkpoints
runpodctl exec <pod-id> -- ls -lh /workspace/models
```

## PyTorch Lightning Configuration

Your `train_batch.py` already uses Lightning correctly:

```python
trainer = Trainer(
    accelerator="gpu",
    devices=1,  # or "auto" for all GPUs
    precision="16-mixed",  # Automatic mixed precision
    max_epochs=50,
    callbacks=[
        ModelCheckpoint(dirpath="/workspace/models", save_top_k=3),
        EarlyStopping(patience=10),
    ],
)
```

## Cost Optimization

### Spot Instances

```bash
runpodctl create pod \
    --gpuType "NVIDIA GeForce RTX 4080 SUPER" \
    --bid 0.3 \
    ...
```

### Stop Immediately After Training

```bash
# In orchestration script
runpodctl exec <pod-id> -- bash -c "..." && \
runpodctl receive <pod-id> /workspace/models ./models && \
runpodctl stop pod <pod-id>
```

## Error Handling

### Check Pod Status

```bash
runpodctl get pod <pod-id>
```

### Validate Training Completion

```bash
# Check for final model
runpodctl exec <pod-id> -- test -f /workspace/models/model_final.pt && echo "Complete"
```

### Resume from Checkpoint

If training is interrupted, Lightning automatically resumes from last checkpoint:

```python
# In train_batch.py - Lightning handles this automatically
trainer.fit(model, datamodule)  # Resumes if checkpoint exists
```

## GPU Selection Guide

| GPU | Cost/hr (spot) | VRAM | Best For |
|-----|----------------|------|----------|
| RTX 3090 | $0.110 | 24GB | Budget training |
| RTX 4080 SUPER | $0.170 | 16GB | Best value |
| RTX 4090 | $0.200 | 24GB | Fastest training |

## Troubleshooting

### Training Not Starting

```bash
# Check pod status
runpodctl get pod <pod-id>

# Check installation logs
runpodctl exec <pod-id> -- cat /workspace/tiny-icf/install.log
```

### Out of Memory

- Reduce batch size: `--batch-size 128`
- Use gradient accumulation: `accumulate_grad_batches=2` in Trainer

### Slow Training

- Increase batch size (if memory allows)
- More data workers: `--num-workers 8`
- Use `precision="16-mixed"` (already default)

## Key Differences: Lightning vs Vanilla

| Task | Lightning | Vanilla |
|------|-----------|---------|
| Checkpointing | `ModelCheckpoint` callback | Manual `torch.save()` |
| Mixed precision | `precision="16-mixed"` | Manual `torch.cuda.amp` |
| Multi-GPU | `devices="auto"` | Manual DDP setup |
| Resume | Automatic | Manual state loading |
| Code lines | ~150 | ~300+ |

## See Also

- `RUNPOD_BEST_PRACTICES.md` - Detailed best practices
- `BATCH_TRAINING_GUIDE.md` - Training configuration guide
- `scripts/runpod_batch.py` - Orchestration script source

