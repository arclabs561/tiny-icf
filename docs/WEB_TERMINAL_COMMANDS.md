# Web Terminal Commands for Monitoring

## Quick Start

Once connected to RunPod web terminal, run these commands:

### 1. Start Ablation Study (One-time)
```bash
cd /root/idf-est
bash scripts/runpod_ablation_oneshot.sh
```

This will:
- Kill any existing processes
- Start ablation study in background
- Save PID to `logs/ablation.pid`
- Redirect all output to `logs/ablation_output.log`

### 2. Monitor Progress
```bash
# Quick status check
bash scripts/runpod_monitor.sh

# Watch live output (updates as it runs)
tail -f logs/ablation_output.log

# Check if still running
ps aux | grep ablation | grep -v grep

# Check GPU usage
watch -n 1 nvidia-smi
```

### 3. Check Results
```bash
# Check if results file exists
ls -lh ablation_results.json

# View results summary
python3 -c "
import json
with open('ablation_results.json') as f:
    d = json.load(f)
print(f'Configs: {len(d)}')
for k, v in d.items():
    print(f'{k}: spearman={v.get(\"final_spearman\", \"N/A\"):.4f}')
"
```

## Detailed Monitoring

### Check Process Status
```bash
# Is it running?
if [ -f logs/ablation.pid ]; then
    PID=$(cat logs/ablation.pid)
    ps -p $PID && echo "Running" || echo "Stopped"
fi
```

### Check Log Progress
```bash
# Count lines (shows progress)
wc -l logs/ablation_output.log

# See last 50 lines
tail -50 logs/ablation_output.log

# Search for errors
grep -i error logs/ablation_output.log | tail -20

# Search for epoch completions
grep -i epoch logs/ablation_output.log | tail -20
```

### Check GPU Usage
```bash
# Current GPU stats
nvidia-smi

# Continuous monitoring (Ctrl+C to stop)
watch -n 1 nvidia-smi
```

### Check Disk Space
```bash
# Check available space
df -h

# Check project size
du -sh /root/idf-est
```

## Troubleshooting

### Process Stopped Unexpectedly
```bash
# Check for errors in log
grep -i "error\|exception\|traceback" logs/ablation_output.log

# Check exit code
tail -5 logs/ablation_status.txt

# Restart if needed
bash scripts/runpod_ablation_oneshot.sh
```

### Out of Memory
```bash
# Check memory usage
free -h
nvidia-smi

# Reduce batch size in script if needed
# Edit: scripts/runpod_ablation_oneshot.sh
# Change: --batch-size 64 to --batch-size 32
```

### Want to Stop
```bash
# Kill the process
if [ -f logs/ablation.pid ]; then
    kill $(cat logs/ablation.pid)
    echo "Process killed"
fi

# Or kill all ablation processes
pkill -f ablation_loss_study
```

## Expected Output

The ablation study will test 5 configurations:
1. `huber_only` - Baseline
2. `pairwise_rank_2.0` - Current pairwise
3. `pairwise_rank_10.0` - High ranking weight
4. `listwise_lambdarank` - LambdaRank
5. `listwise_approx_ndcg` - ApproxNDCG
6. `diffsort` - Differentiable sorting (if available)

Each runs for 15 epochs. Total time: ~1-3 hours depending on GPU.

## Completion

When done, you'll see:
- `ablation_results.json` with all results
- Final status in `logs/ablation_status.txt`
- Complete log in `logs/ablation_output.log`

Download results:
```bash
# From local machine
just download
```

