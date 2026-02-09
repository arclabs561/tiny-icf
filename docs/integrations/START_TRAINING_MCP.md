# Start Training - Both MCP Methods

## Method 1: Direct MCP Tool Execution (In Cursor)

Since MCP RunPod tools are configured in Cursor, you can execute commands directly:

### Step 1: Execute Training Command

**In Cursor, use the MCP RunPod tools:**

```
Execute on RunPod pod 9lj0lizlogeftc:
cd /workspace/tiny-icf && bash start_mcp.sh
```

Or with direct uv execution:
```
Execute on RunPod pod 9lj0lizlogeftc:
cd /workspace/tiny-icf && uv run auto_train_mcp.py
```

### Step 2: Monitor Training

```
Show the last 20 lines of /workspace/tiny-icf/training.log from RunPod pod 9lj0lizlogeftc
```

Or continuous monitoring:
```
Monitor training progress on RunPod pod 9lj0lizlogeftc by tailing /workspace/tiny-icf/training.log
```

### Step 3: Check Status

```
Check status of RunPod pod 9lj0lizlogeftc
```

## Method 2: RunPod API Direct (Programmatic)

The script `scripts/execute_via_mcp_api.py` attempts to use the RunPod GraphQL API directly:

```bash
uv run scripts/execute_via_mcp_api.py 9lj0lizlogeftc
```

This tries multiple GraphQL mutations to find one that works for command execution.

## Method 3: Manual SSH (Fallback)

If MCP tools don't support command execution:

```bash
runpodctl ssh connect 9lj0lizlogeftc
cd /workspace/tiny-icf && bash start_mcp.sh
```

## 📊 Current Status

- ✅ Pod: `9lj0lizlogeftc` (RUNNING)
- ✅ Scripts: Uploaded and ready
- ✅ Training script: PEP 723 compliant
- ✅ Dependencies: Managed by `uv run`

## 🎯 Training Configuration

- **Epochs**: 50
- **Batch Size**: 256
- **Precision**: 16-mixed (Lightning)
- **Devices**: 1 GPU
- **Output**: `/workspace/tiny-icf/models/`

## 📥 Download Results

After training completes:
```
Download /workspace/tiny-icf/models/ from RunPod pod 9lj0lizlogeftc to ./models/
```

Or use runpodctl:
```bash
uv run scripts/runpod_batch.py --download-only --pod-id 9lj0lizlogeftc
```

