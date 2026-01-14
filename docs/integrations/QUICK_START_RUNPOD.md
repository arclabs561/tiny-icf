# Quick Start: RunPod Training

## One-Command Setup

```bash
# From local machine
just deploy
```

This will:
1. Sync all code to RunPod
2. Install dependencies (uv, diffsort, torch, etc.)
3. Set up environment

## Start Training

### Option 1: Ablation Study (Compare All Loss Methods)

**From local machine:**
```bash
just ablation
```

**Or in web terminal:**
```bash
cd /root/idf-est
bash scripts/runpod_ablation_oneshot.sh
```

### Option 2: Train with Best Practices

**From local machine:**
```bash
just train-best
```

**With Aim tracking:**
```bash
just train-best-aim
```

**Or in web terminal:**
```bash
cd /root/idf-est
export PATH="$HOME/.cargo/bin:$PATH"
uv run scripts/train_best_practices.py \
    --data data/word_frequency.csv \
    --epochs 100 \
    --batch-size 64 \
    --output models/model_best.pt \
    --early-stop \
    --aim \
    --aim-experiment "runpod-training"
```

## Monitor Progress

### In Web Terminal

```bash
# Quick status
bash scripts/runpod_monitor.sh

# Watch live output
tail -f logs/ablation_output.log

# Check GPU
watch -n 1 nvidia-smi
```

### From Local Machine

```bash
# Check status
just monitor

# Download results when done
just download
```

## Aim Tracking

### Start Aim UI on RunPod

**In web terminal:**
```bash
cd /root/idf-est
bash scripts/runpod_start_aim.sh
```

**Access from local (with SSH tunnel):**
```bash
# In one terminal - start tunnel
ssh -p 31179 -i ~/.ssh/id_ed25519 -L 43800:localhost:43800 root@38.80.152.76

# In another terminal - start Aim
just aim-remote

# Or use justfile command
just aim-remote
```

Then open: http://127.0.0.1:43800

### View Experiments

Aim will show:
- Training metrics (loss, Spearman, MAE, RBO)
- Hyperparameters
- System metrics (GPU usage, memory)
- Model checkpoints
- Comparison across experiments

## Expected Results

### Ablation Study
Tests 5-6 configurations:
1. `huber_only` - Baseline
2. `pairwise_rank_2.0` - Current pairwise
3. `pairwise_rank_10.0` - High ranking weight
4. `listwise_lambdarank` - LambdaRank
5. `listwise_approx_ndcg` - ApproxNDCG
6. `diffsort` - Differentiable sorting

**Output:** `ablation_results.json` with all metrics

### Best Practices Training
- All improvements enabled
- Early stopping
- Aim tracking
- Best model saved

**Output:** `models/model_best.pt` + Aim experiments

## Troubleshooting

### Connection Refused
- Check RunPod console for pod status
- Get new IP/port if pod restarted
- Update `justfile` with new connection details

### Process Hanging
- Use web terminal instead of SSH
- Check with `bash scripts/runpod_monitor.sh`
- Kill and restart: `pkill -f ablation_loss_study`

### Out of Memory
- Reduce batch size: `--batch-size 32`
- Or use smaller dataset subset

### Results Not Appearing
- Check if process is still running: `ps aux | grep ablation`
- Check logs: `cat logs/ablation_output.log`
- Check for errors: `grep -i error logs/ablation_output.log`

## Next Steps

1. **Run ablation study** to compare loss methods
2. **Train best method** with full epochs
3. **Evaluate on Jabberwocky Protocol**
4. **Analyze results** and iterate

All commands are ready - just sync and run!

