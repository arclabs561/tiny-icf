# Start Training on Pod

## Current Status

✅ **Pod**: `9lj0lizlogeftc` (RUNNING)  
✅ **Project uploaded** to `/workspace/tiny-icf`  
✅ **Training script** ready at `/workspace/tiny-icf/start_training.sh`

## Start Training

### Option 1: SSH and Run (Recommended)

```bash
# Connect to pod
runpodctl ssh connect 9lj0lizlogeftc

# Inside pod, start training:
cd /workspace/tiny-icf
bash start_training.sh

# Monitor (in another terminal or after detaching):
tail -f /workspace/tiny-icf/training.log
```

### Option 2: Check if Already Running

```bash
runpodctl ssh connect 9lj0lizlogeftc
ps aux | grep train
cat /workspace/tiny-icf/training.pid 2>/dev/null
```

## Training Configuration

- **Epochs**: 50
- **Batch Size**: 256
- **Devices**: 1 GPU
- **Data**: Multilingual combined
- **Output**: `/workspace/tiny-icf/models/`

## Monitor Progress

```bash
# SSH to pod
runpodctl ssh connect 9lj0lizlogeftc

# Watch log
tail -f /workspace/tiny-icf/training.log

# Check process
ps aux | grep train
```

## Download Results

When training completes:

```bash
uv run scripts/runpod_batch.py --download-only --pod-id 9lj0lizlogeftc
```

Or manually:

```bash
runpodctl receive 9lj0lizlogeftc /workspace/tiny-icf/models ./models
```

