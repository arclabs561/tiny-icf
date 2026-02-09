# RunPod Setup and Training Guide

## SSH Connection

```bash
ssh root@205.196.19.18 -p 11859 -i ~/.ssh/id_ed25519
```

**Note**: If SSH fails with "Permission denied", verify:
1. The SSH key is added to the RunPod instance (check RunPod dashboard → SSH Keys)
2. The connection details are correct (IP, port, username)
3. The key file permissions: `chmod 600 ~/.ssh/id_ed25519`

## Quick Setup

Once connected to RunPod, run:

```bash
# Option 1: Run the setup script
bash <(curl -s https://raw.githubusercontent.com/arclabs561/tiny-icf/main/scripts/runpod_setup.sh)

# Option 2: Manual setup
cd /root/idf-est || git clone https://github.com/arclabs561/tiny-icf.git idf-est && cd idf-est
export PATH="$HOME/.cargo/bin:$PATH"
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install diffsort torch numpy pandas tqdm scipy lightning
```

## Running Experiments

### 1. Ablation Study (Compare All Loss Methods)

Compares all loss configurations including differentiable sorting:

```bash
cd /root/idf-est
export PATH="$HOME/.cargo/bin:$PATH"
uv run scripts/ablation_loss_study.py \
    --data data/word_frequency.csv \
    --epochs 15 \
    --batch-size 64 \
    --output ablation_results.json \
    2>&1 | tee ablation_run.log
```

This will test:
- Huber loss only
- Pairwise ranking (weight=2.0, 10.0)
- Listwise LambdaRank
- Listwise ApproxNDCG
- Differentiable sorting (diffsort, if available)

### 2. Train with Differentiable Sorting

Direct Spearman correlation optimization:

```bash
uv run scripts/train_diffsort.py \
    --data data/word_frequency.csv \
    --epochs 50 \
    --batch-size 64 \
    --method diffsort \
    --huber-weight 0.3 \
    --output models/model_diffsort.pt \
    --history training_history.json \
    2>&1 | tee train_diffsort.log
```

### 3. Train with Listwise Loss

Listwise ranking optimization:

```bash
uv run scripts/train_listwise.py \
    --data data/word_frequency.csv \
    --epochs 50 \
    --batch-size 64 \
    --listwise-method lambdarank \
    --output models/model_listwise.pt \
    2>&1 | tee train_listwise.log
```

### 4. Best Practices Training

Unified training with all improvements:

```bash
uv run scripts/train_best_practices.py \
    --data data/word_frequency.csv \
    --epochs 100 \
    --batch-size 64 \
    --output models/model_best.pt \
    --early-stop \
    --early-stop-patience 15 \
    2>&1 | tee train_best.log
```

## Monitoring Training

### Real-time Monitoring

```bash
# Watch log file
tail -f train_diffsort.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check process
ps aux | grep python
```

### Download Results

From your local machine:

```bash
# Download model
scp -P 11859 -i ~/.ssh/id_ed25519 \
    root@205.196.19.18:/root/idf-est/models/model_diffsort.pt \
    ./models/

# Download results
scp -P 11859 -i ~/.ssh/id_ed25519 \
    root@205.196.19.18:/root/idf-est/ablation_results.json \
    ./

# Download logs
scp -P 11859 -i ~/.ssh/id_ed25519 \
    root@205.196.19.18:/root/idf-est/*.log \
    ./
```

## Background Training

To run training in the background (survives SSH disconnect):

```bash
# Using nohup
nohup uv run scripts/train_diffsort.py \
    --data data/word_frequency.csv \
    --epochs 50 \
    --batch-size 64 \
    --output models/model_diffsort.pt \
    > train_diffsort.log 2>&1 &

# Check background job
jobs
fg  # Bring to foreground
bg  # Send to background

# Or use screen/tmux
screen -S training
# Run training command
# Detach: Ctrl+A, D
# Reattach: screen -r training
```

## Troubleshooting

### Permission Denied (SSH)

1. Check key is added to RunPod: Dashboard → SSH Keys
2. Verify key file: `ls -la ~/.ssh/id_ed25519`
3. Try with verbose: `ssh -v root@205.196.19.18 -p 11859 -i ~/.ssh/id_ed25519`

### Module Not Found

```bash
# Reinstall dependencies
export PATH="$HOME/.cargo/bin:$PATH"
uv pip install --upgrade diffsort torch numpy pandas tqdm scipy
```

### CUDA Out of Memory

Reduce batch size:
```bash
--batch-size 32  # or 16, 8
```

### Data File Missing

```bash
# Download data
python scripts/download_data.py

# Or manually download
wget https://raw.githubusercontent.com/arclabs561/tiny-icf/main/data/word_frequency.csv -O data/word_frequency.csv
```

## Expected Results

After running the ablation study, you should see:

- **Huber only**: Baseline, no ranking signal
- **Pairwise ranking**: Moderate Spearman improvement
- **Listwise losses**: Better ranking optimization
- **Differentiable sorting**: Direct Spearman optimization (should be best)

Target metrics:
- **Spearman correlation**: > 0.5 (good), > 0.7 (excellent)
- **MAE**: < 0.15
- **RBO (full)**: > 0.3

## Next Steps

1. Run ablation study to compare all methods
2. Train best-performing method for full epochs
3. Evaluate on Jabberwocky Protocol
4. Compare with baseline models

