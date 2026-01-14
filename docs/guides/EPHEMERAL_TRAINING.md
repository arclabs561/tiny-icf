# Ephemeral Training Setup

## Overview

This setup is designed for training on ephemeral RunPod environments that may restart or change frequently. The training infrastructure handles pod restarts gracefully with automatic checkpointing and resume.

## Files Created

### 1. `scripts/train_ephemeral_robust.py`
Robust training script with:
- **Graceful shutdown handling**: Catches SIGTERM/SIGINT and saves checkpoint before exit
- **Frequent checkpointing**: Saves checkpoint every epoch (configurable)
- **Auto-resume**: Automatically resumes from checkpoint if found
- **Error recovery**: Continues training even if individual epochs fail
- **Persistent directory support**: Can use persistent storage if available

### 2. `scripts/run_ephemeral_training.sh`
Management script with commands:
- `start` - Start new training session
- `resume` - Resume from checkpoint
- `monitor` - Check training status
- `stop` - Stop training gracefully
- `logs` - Follow training logs

### 3. `scripts/monitor_ephemeral.sh`
Quick monitoring script to check:
- Training process status
- Recent logs
- Checkpoint status
- Model files

## Usage

### Initial Setup (on remote server)

```bash
# SSH to server
ssh -i ~/.ssh/id_ed25519 -p 37707 root@213.173.111.79

# Navigate to project
cd /root/idf-est

# Install dependencies (if needed)
apt-get update
apt-get install -y python3-pip python3-venv python3-dev
pip3 install --root-user-action=ignore torch numpy pandas tqdm scipy

# Verify installation
python3 -c "import torch; print('PyTorch ready')"
```

### Starting Training

```bash
# On remote server
cd /root/idf-est
python3 scripts/train_ephemeral_robust.py \
    --data data/word_frequency.csv \
    --output-dir models \
    --epochs 200 \
    --batch-size 256 \
    --lr 1e-3 \
    --rank-weight 5.0 \
    --early-stop-patience 20 \
    --checkpoint-interval 1
```

Or use the management script:
```bash
./scripts/run_ephemeral_training.sh start
```

### Resuming After Pod Restart

When the pod restarts, simply run the same command - it will automatically detect and resume from the checkpoint:

```bash
# Auto-resumes from models/checkpoint_ephemeral_robust.pt
python3 scripts/train_ephemeral_robust.py \
    --data data/word_frequency.csv \
    --output-dir models \
    --epochs 200 \
    --batch-size 256 \
    --lr 1e-3 \
    --rank-weight 5.0 \
    --early-stop-patience 20 \
    --checkpoint-interval 1
```

Or explicitly specify checkpoint:
```bash
python3 scripts/train_ephemeral_robust.py \
    ... \
    --resume models/checkpoint_ephemeral_robust.pt
```

### Monitoring

From local machine:
```bash
./scripts/monitor_ephemeral.sh
```

Or SSH and check directly:
```bash
ssh -i ~/.ssh/id_ed25519 -p 37707 root@213.173.111.79
cd /root/idf-est
tail -f training_ephemeral.log
```

## Features for Ephemeral Pods

### 1. Checkpoint Every Epoch
- Saves full training state after each epoch
- Includes: model, optimizer, scheduler, epoch, history, best model
- Can resume from any epoch

### 2. Graceful Shutdown
- Handles SIGTERM (pod shutdown) and SIGINT (Ctrl+C)
- Saves checkpoint before exiting
- Ensures no data loss on pod restart

### 3. Auto-Resume
- Automatically detects checkpoint on startup
- Resumes from exact epoch where training stopped
- Preserves optimizer state, learning rate schedule, and history

### 4. Error Recovery
- If an epoch fails, saves checkpoint and continues
- Prevents losing progress due to transient errors
- Logs errors for debugging

### 5. Persistent Storage Support
- Can use `--persistent-dir` for persistent storage
- Falls back to local directory if persistent storage unavailable
- Checkpoints work in both scenarios

## Checkpoint Structure

Checkpoints contain:
- `epoch`: Current epoch number
- `model_state_dict`: Full model weights
- `optimizer_state_dict`: Optimizer state (momentum, etc.)
- `scheduler_state_dict`: Learning rate scheduler state
- `best_spearman`: Best validation score so far
- `best_model_state`: Best model weights
- `history`: Training history (loss, metrics per epoch)
- `args`: Training arguments

## Best Practices

1. **Frequent Checkpoints**: Use `--checkpoint-interval 1` for maximum safety
2. **Monitor Regularly**: Check training status periodically
3. **Save Best Models**: Best models are saved separately for evaluation
4. **Log Everything**: All output goes to `training_ephemeral.log`
5. **Use nohup**: Run with `nohup` to survive SSH disconnects

## Troubleshooting

### Pod Connection Refused
- Pod may have restarted (ephemeral nature)
- Wait for pod to come back online
- Resume from checkpoint when pod is available

### Missing Dependencies
```bash
pip3 install --root-user-action=ignore torch numpy pandas tqdm scipy
```

### Checkpoint Not Found
- Check if `models/checkpoint_ephemeral_robust.pt` exists
- If missing, training will start from scratch
- Previous checkpoints may be lost if pod storage is ephemeral

### Training Process Not Found
- Process may have completed or crashed
- Check logs: `tail -f training_ephemeral.log`
- Resume from checkpoint if needed

## Example Workflow

1. **Start Training**:
   ```bash
   nohup python3 scripts/train_ephemeral_robust.py ... > training.log 2>&1 &
   ```

2. **Pod Restarts** (ephemeral nature)

3. **Resume Training**:
   ```bash
   # Same command - auto-resumes
   nohup python3 scripts/train_ephemeral_robust.py ... > training.log 2>&1 &
   ```

4. **Monitor Progress**:
   ```bash
   tail -f training_ephemeral.log
   ```

5. **Check Status**:
   ```bash
   ./scripts/monitor_ephemeral.sh
   ```

## Notes

- Pod IP/port may change - update SSH connection details as needed
- Checkpoints are saved locally in `models/` directory
- For true persistence, use `--persistent-dir` with mounted storage
- Training can be interrupted and resumed multiple times
- Best model is saved separately for final evaluation

