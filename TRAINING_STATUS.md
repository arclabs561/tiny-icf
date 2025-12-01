# Training Status & Quick Start

## ✅ Setup Complete

**Pod**: `9lj0lizlogeftc` (RUNNING - RTX 3090)  
**Scripts Uploaded**: ✅  
- `/workspace/tiny-icf/auto_train.py` - PEP 723 training script  
- `/workspace/tiny-icf/start.sh` - Bash starter  

## 🚀 Start Training (One Command)

```bash
runpodctl ssh connect 9lj0lizlogeftc
cd /workspace/tiny-icf && bash start.sh
```

Or if you want to see output immediately:

```bash
runpodctl ssh connect 9lj0lizlogeftc
cd /workspace/tiny-icf && uv run auto_train.py
```

## 📊 Monitor Training

```bash
# SSH to pod
runpodctl ssh connect 9lj0lizlogeftc

# Watch log
tail -f /workspace/tiny-icf/training.log

# Check if running
cat /workspace/tiny-icf/training.pid
ps aux | grep train
```

## 📥 Download Results

When training completes:

```bash
uv run scripts/runpod_batch.py --download-only --pod-id 9lj0lizlogeftc
```

## 🔧 What's Ready

- **PEP 723 Script**: `auto_train.py` - Self-contained, uses `uv run`
- **Starter**: `start.sh` - Runs training in background
- **Training Config**:
  - Epochs: 50
  - Batch size: 256
  - Mixed precision: 16-mixed
  - Devices: 1 GPU
  - Output: `/workspace/tiny-icf/models/`

## ⚡ Quick Commands

```bash
# Check status
runpodctl ssh connect 9lj0lizlogeftc -c "cat /workspace/tiny-icf/training.pid 2>/dev/null || echo 'Not started'"

# View last 20 log lines
runpodctl ssh connect 9lj0lizlogeftc -c "tail -20 /workspace/tiny-icf/training.log 2>/dev/null || echo 'No log'"
```
