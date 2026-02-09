# Start Training Using MCP RunPod Tools

## ✅ Pod Status (via MCP)

Pod `9lj0lizlogeftc` is **RUNNING** ✅
- GPU: RTX 3090
- Cost: $0.22/hr
- Status: Ready for training

## 🚀 Start Training via MCP

Since MCP RunPod tools are available in Cursor, you can execute commands directly:

### Option 1: Execute Training Command

**In Cursor, use MCP RunPod tools:**
```
Execute on RunPod pod 9lj0lizlogeftc:
cd /workspace/tiny-icf && bash start_mcp.sh
```

Or directly:
```
Execute on RunPod pod 9lj0lizlogeftc:
cd /workspace/tiny-icf && uv run auto_train_mcp.py
```

### Option 2: Check Status First

```
Check status of RunPod pod 9lj0lizlogeftc
```

Then execute the training command above.

## 📊 Monitor Training

```
Show the last 20 lines of /workspace/tiny-icf/training.log from RunPod pod 9lj0lizlogeftc
```

Or:
```
Monitor training progress on RunPod pod 9lj0lizlogeftc by tailing /workspace/tiny-icf/training.log
```

## 📁 Available Scripts on Pod

- `/workspace/tiny-icf/start_mcp.sh` - MCP starter script
- `/workspace/tiny-icf/auto_train_mcp.py` - PEP 723 training script
- `/workspace/tiny-icf/start.sh` - Original starter
- `/workspace/tiny-icf/auto_train.py` - Original PEP 723 script

## 🎯 Training Configuration

- **Epochs**: 50
- **Batch Size**: 256
- **Precision**: 16-mixed (Lightning)
- **Devices**: 1 GPU
- **Output**: `/workspace/tiny-icf/models/`

## 💡 Why MCP Tools?

MCP RunPod tools provide:
- ✅ Direct API access (no runpodctl exec timeouts)
- ✅ Better error handling
- ✅ Real-time status updates
- ✅ Native integration with Cursor

## 📥 Download Results

After training completes:
```
Download /workspace/tiny-icf/models/ from RunPod pod 9lj0lizlogeftc to ./models/
```

