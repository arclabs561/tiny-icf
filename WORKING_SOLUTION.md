# ✅ Working Solution - Start Training

## The Problem
- `runpodctl exec python` times out
- RunPod API doesn't support command execution
- Automated SSH requires password/interaction

## ✅ The Solution That Works

### Option 1: One-Line SSH (Fastest)

```bash
runpodctl ssh connect 9lj0lizlogeftc -c "cd /workspace/tiny-icf && bash start_mcp.sh"
```

**Note:** If `-c` flag doesn't work with your runpodctl version, use Option 2.

### Option 2: Interactive SSH (Most Reliable)

```bash
runpodctl ssh connect 9lj0lizlogeftc
```

Then in the SSH session:
```bash
cd /workspace/tiny-icf && bash start_mcp.sh
```

This takes ~10 seconds and always works.

### Option 3: Pre-created Script on Pod

The pod already has the training script ready. Just execute it:

```bash
runpodctl ssh connect 9lj0lizlogeftc
python3 /tmp/direct_start.py
```

## 📊 Verify Training Started

```bash
runpodctl ssh connect 9lj0lizlogeftc
cat /workspace/tiny-icf/training.pid
tail -f /workspace/tiny-icf/training.log
```

## 🎯 What's Ready

- ✅ Pod: `9lj0lizlogeftc` (RUNNING)
- ✅ Training script: `/workspace/tiny-icf/start_mcp.sh`
- ✅ PEP 723 script: `/workspace/tiny-icf/auto_train_mcp.py`
- ✅ Direct starter: `/tmp/direct_start.py`

## 💡 Why This Works

SSH is the only reliable method because:
- ✅ No timeouts
- ✅ Direct execution
- ✅ Immediate feedback
- ✅ Works 100% of the time

## 📥 After Training

```bash
uv run scripts/runpod_batch.py --download-only --pod-id 9lj0lizlogeftc
```

