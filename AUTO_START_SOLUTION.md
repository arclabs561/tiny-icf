# Auto-Start Training Solution

## Issue
`runpodctl exec python` consistently times out, making automatic execution unreliable.

## ✅ Working Solutions

### Solution 1: SSH One-Liner (Most Reliable)
```bash
runpodctl ssh connect 9lj0lizlogeftc -c 'cd /workspace/tiny-icf && bash start.sh'
```

### Solution 2: Interactive SSH
```bash
runpodctl ssh connect 9lj0lizlogeftc
cd /workspace/tiny-icf && bash start.sh
```

### Solution 3: Pre-uploaded Script Execution
All scripts are already on the pod:
- `/workspace/tiny-icf/start.sh` - Main starter
- `/workspace/tiny-icf/auto_train.py` - PEP 723 training script
- `/workspace/tiny-icf/start_simple.sh` - Simple starter
- `/tmp/api_start.py` - API-based starter

Execute any of these via SSH.

## 📊 What's Ready

✅ **Pod**: `9lj0lizlogeftc` (RUNNING)  
✅ **Scripts**: All uploaded  
✅ **PEP 723**: Training script uses `uv run`  
✅ **Background**: All scripts use `nohup` for background execution  

## 🚀 Quick Start

```bash
# One command to start training
runpodctl ssh connect 9lj0lizlogeftc -c 'cd /workspace/tiny-icf && bash start.sh'

# Monitor
runpodctl ssh connect 9lj0lizlogeftc -c 'tail -f /workspace/tiny-icf/training.log'
```

## 🔧 Why Auto-Start Failed

`runpodctl exec python` has a timeout issue - it hangs or times out before commands complete. This is a limitation of the runpodctl tool, not our scripts.

## 💡 Best Practice

For reliable execution, use SSH directly:
- Fast
- Reliable  
- No timeouts
- Can see output immediately

## 📥 Download Results

```bash
uv run scripts/runpod_batch.py --download-only --pod-id 9lj0lizlogeftc
```

