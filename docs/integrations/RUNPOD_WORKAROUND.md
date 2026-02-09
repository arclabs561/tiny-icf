# RunPod Non-Interactive Execution: Workarounds

## The Problem

`runpodctl exec` can have issues:
- Timeouts on long-running commands
- Connection failures
- Limited error handling
- No reliable background execution

## The Solution: Multi-Layer Approach

We use **three methods** in order of preference:

### Method 1: `runpodctl exec` (Primary)

Direct command execution - fastest when it works:

```bash
runpodctl exec <pod-id> -- bash -c "cd /workspace/tiny-icf && uv run scripts/train_batch.py ..."
```

**Pros:**
- Simple, direct
- Immediate feedback
- Works for short commands

**Cons:**
- Can timeout on long commands
- Connection issues
- Background execution unreliable

### Method 2: Auto-Start Script (Fallback)

Upload a script that runs training, then trigger it:

```bash
# Create auto-start script
uv run scripts/runpod_auto_start.py <pod-id>

# Script is now at /workspace/tiny-icf/auto_start_training.sh
# Trigger it
runpodctl exec <pod-id> -- bash /workspace/tiny-icf/auto_start_training.sh
```

**Pros:**
- More reliable than complex commands
- Script can be reused
- Handles background execution better
- Can check if already running

**Cons:**
- Requires two steps (upload + trigger)
- Still uses `runpodctl exec` to trigger

### Method 3: RunPod GraphQL API (Future)

Direct API calls bypass `runpodctl` entirely:

```python
# Use scripts/runpod_api_execute.py
uv run scripts/runpod_api_execute.py <pod-id> --command "cd /workspace/tiny-icf && ..."
```

**Pros:**
- Bypasses `runpodctl` limitations
- Direct API access
- Better error handling
- Async execution support

**Cons:**
- Requires API key
- More complex implementation
- API schema may change

## Recommended Workflow

### For Background Training (Recommended)

```bash
# 1. Upload project
uv run scripts/runpod_batch.py --upload-only --pod-id <pod-id>

# 2. Start training (uses auto-start fallback)
uv run scripts/runpod_batch.py --background --pod-id <pod-id>

# 3. Check status
uv run scripts/runpod_auto_start.py <pod-id> --status

# 4. Monitor logs
runpodctl exec <pod-id> -- tail -f /workspace/tiny-icf/training.log

# 5. Download when complete
uv run scripts/runpod_batch.py --download-only --pod-id <pod-id>
```

### For Foreground Training

```bash
# Single command (waits for completion)
uv run scripts/runpod_batch.py --pod-id <pod-id>
```

This will:
1. Try `runpodctl exec` first
2. Fall back to auto-start script if exec fails
3. Provide manual instructions if both fail

## Auto-Start Script Features

The auto-start script (`auto_start_training.sh`) includes:

- ✅ **Idempotency**: Checks if training already running
- ✅ **Completion check**: Skips if `model_final.pt` exists
- ✅ **PID tracking**: Saves PID to `training.pid`
- ✅ **Logging**: All output to `training.log`
- ✅ **Background execution**: Uses `nohup` for reliability

## Status Checking

Check training status without SSH:

```bash
# Using auto-start script
uv run scripts/runpod_auto_start.py <pod-id> --status

# Or manually
runpodctl exec <pod-id> -- bash -c "
if [ -f /workspace/tiny-icf/training.pid ]; then
  PID=\$(cat /workspace/tiny-icf/training.pid)
  if ps -p \$PID > /dev/null 2>&1; then
    echo 'RUNNING'
  else
    echo 'STOPPED'
  fi
else
  echo 'NOT_STARTED'
fi
"
```

## Why This Works

1. **Multiple fallbacks**: If one method fails, try the next
2. **Idempotent scripts**: Can be run multiple times safely
3. **Background execution**: Uses `nohup` for reliability
4. **Status tracking**: PID files and completion checks
5. **No SSH required**: All methods use non-interactive commands

## Best Practice

**Always use the auto-start script method for production:**

```bash
# One-time setup
uv run scripts/runpod_auto_start.py <pod-id>

# Then trigger whenever needed
runpodctl exec <pod-id> -- bash /workspace/tiny-icf/auto_start_training.sh
```

This is the most reliable method because:
- Script handles edge cases (already running, already complete)
- Uses `nohup` for true background execution
- Can be triggered multiple times safely
- Works even if `runpodctl exec` has connection issues

## Troubleshooting

### Training Not Starting

```bash
# Check if script exists
runpodctl exec <pod-id> -- test -f /workspace/tiny-icf/auto_start_training.sh && echo "Script exists"

# Check pod status
runpodctl get pod <pod-id>

# Try manual trigger
runpodctl exec <pod-id> -- bash /workspace/tiny-icf/auto_start_training.sh
```

### Training Stuck

```bash
# Check if process is running
runpodctl exec <pod-id> -- ps aux | grep train_batch

# Check logs
runpodctl exec <pod-id> -- tail -n 50 /workspace/tiny-icf/training.log

# Kill if needed
runpodctl exec <pod-id> -- pkill -f train_batch
```

### Connection Issues

If `runpodctl exec` consistently fails:
1. Use auto-start script (more reliable)
2. Fall back to SSH for manual execution
3. Consider using RunPod API directly (future)

## Summary

**The workaround is simple**: Use an auto-start script that handles all edge cases, then trigger it with `runpodctl exec`. This is more reliable than complex inline commands and provides better error handling.

The `runpod_batch.py` script automatically tries this fallback if direct execution fails, so you get the best of both worlds: fast direct execution when it works, reliable fallback when it doesn't.

