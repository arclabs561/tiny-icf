# RunPod API - Practical Usage Guide

## Quick Start

### 1. Monitor Existing Pod
```bash
# Real-time monitoring
uv run scripts/runpod_monitor.py <pod-id> --interval 10 --duration 300

# One-time status check
uv run scripts/runpod_api_client.py get <pod-id>
```

### 2. List All Pods
```bash
uv run scripts/runpod_api_client.py list
```

### 3. Create Pod via API
```bash
# Using batch script (recommended)
uv run scripts/runpod_batch.py --use-api --upload-only

# Or using API client directly
uv run scripts/runpod_api_client.py create \
    --name "my-pod" \
    --gpu-type "NVIDIA GeForce RTX 4080 SUPER" \
    --image "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
    --gpu-count 1 \
    --volume-gb 50
```

### 4. Start/Stop Pod
```bash
# Start pod
uv run scripts/runpod_api_client.py start <pod-id>

# Stop pod
uv run scripts/runpod_api_client.py stop <pod-id>
```

## Real-World Workflows

### Workflow 1: Create and Train (API + runpodctl)

```bash
# Step 1: Create pod via API
uv run scripts/runpod_batch.py --use-api --upload-only

# Step 2: Training starts automatically (or manually)
uv run scripts/runpod_batch.py --pod-id <pod-id> --background

# Step 3: Monitor via API (better than runpodctl)
uv run scripts/runpod_monitor.py <pod-id> --interval 30

# Step 4: Check status
uv run scripts/runpod_batch.py --pod-id <pod-id> --status

# Step 5: Download results (uses runpodctl)
uv run scripts/runpod_batch.py --pod-id <pod-id> --download-only
```

### Workflow 2: Monitor Training Progress

```bash
# Start monitoring in background
uv run scripts/runpod_monitor.py <pod-id> --interval 10 --duration 3600 &

# Or check periodically
watch -n 30 "uv run scripts/runpod_api_client.py get <pod-id> | jq '.runtime.gpus[0].gpuUtilPercent'"
```

### Workflow 3: Automated Pod Management

```python
from scripts.runpod_api_client import RunPodAPIClient
import time

client = RunPodAPIClient()

# Create pod
pod = client.create_pod(
    name="auto-training",
    gpu_type_id="NVIDIA GeForce RTX 4080 SUPER",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_count=1,
    volume_in_gb=50,
)
pod_id = pod['id']

# Wait for pod to be ready
while True:
    pod_info = client.get_pod(pod_id)
    if pod_info['desiredStatus'] == 'RUNNING':
        break
    time.sleep(10)

# Monitor training
while True:
    pod_info = client.get_pod(pod_id)
    gpus = pod_info.get('runtime', {}).get('gpus', [])
    if gpus:
        util = gpus[0].get('gpuUtilPercent', 0)
        print(f"GPU Utilization: {util}%")
        
        if util == 0:  # Training finished
            break
    time.sleep(60)

# Stop pod when done
client.stop_pod(pod_id)
```

## API vs runpodctl Comparison

| Operation | API | runpodctl | Best Choice |
|-----------|-----|-----------|-------------|
| Create pod | ✅ | ✅ | **API** (programmatic) |
| List pods | ✅ | ✅ | **API** (structured data) |
| Get pod info | ✅ | ✅ | **API** (better telemetry) |
| Start/stop pod | ✅ | ✅ | **API** (programmatic) |
| Upload files | ❌ | ✅ | **runpodctl** (only option) |
| Execute commands | ❌ | ✅ | **runpodctl** (only option) |
| SSH connection | ❌ | ✅ | **runpodctl** (only option) |

## Monitoring Best Practices

### Real-Time Monitoring
```bash
# Monitor with 10s updates for 5 minutes
uv run scripts/runpod_monitor.py <pod-id> --interval 10 --duration 300
```

### Periodic Status Checks
```bash
# Check every 30 seconds
watch -n 30 "uv run scripts/runpod_api_client.py get <pod-id> | jq '.runtime'"
```

### GPU Utilization Tracking
```bash
# Track GPU usage over time
while true; do
    uv run scripts/runpod_api_client.py get <pod-id> | \
        jq -r '.runtime.gpus[0] | "\(.gpuUtilPercent)% util, \(.memoryUtilPercent)% mem"'
    sleep 30
done
```

## Cost Monitoring

```bash
# Get current cost
uv run scripts/runpod_api_client.py get <pod-id> | jq '.costPerHr'

# Calculate estimated cost
pod_info=$(uv run scripts/runpod_api_client.py get <pod-id>)
cost_per_hr=$(echo "$pod_info" | jq '.costPerHr')
uptime=$(echo "$pod_info" | jq '.runtime.uptimeInSeconds')
hours=$(echo "scale=2; $uptime / 3600" | bc)
total_cost=$(echo "scale=4; $cost_per_hr * $hours" | bc)
echo "Estimated cost so far: \$$total_cost"
```

## Error Handling

The API client provides clear error messages:

```python
from scripts.runpod_api_client import RunPodAPIClient, RunPodAPIError

try:
    client = RunPodAPIClient()
    pod = client.get_pod("invalid_id")
except RunPodAPIError as e:
    print(f"API Error: {e}")  # Clear error message
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Tips and Tricks

### 1. Find Best GPU for Your Budget
```bash
# List all GPUs with prices
uv run scripts/runpod_api_client.py gpu-types | \
    grep -E "(id|displayName|securePrice)" | \
    head -20
```

### 2. Check Pod Health
```bash
# Quick health check
pod_info=$(uv run scripts/runpod_api_client.py get <pod-id>)
status=$(echo "$pod_info" | jq -r '.desiredStatus')
uptime=$(echo "$pod_info" | jq '.runtime.uptimeInSeconds')
echo "Status: $status, Uptime: $((uptime / 3600))h"
```

### 3. Auto-Stop Idle Pods
```python
# Stop pods that have been idle for > 1 hour
pods = client.list_pods()
for pod in pods:
    if pod['desiredStatus'] == 'RUNNING':
        info = client.get_pod(pod['id'])
        runtime = info.get('runtime', {})
        gpus = runtime.get('gpus', [])
        
        if gpus and all(g.get('gpuUtilPercent', 0) < 5 for g in gpus):
            uptime = runtime.get('uptimeInSeconds', 0)
            if uptime > 3600:  # Idle for > 1 hour
                client.stop_pod(pod['id'])
                print(f"Stopped idle pod: {pod['id']}")
```

## Integration Examples

### With CI/CD
```yaml
# GitHub Actions example
- name: Create RunPod
  run: |
    uv run scripts/runpod_api_client.py create \
      --name "ci-training-${{ github.run_id }}" \
      --gpu-type "NVIDIA GeForce RTX 4080 SUPER" \
      --image "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

- name: Monitor Training
  run: |
    uv run scripts/runpod_monitor.py $POD_ID --duration 3600
```

### With Python Scripts
```python
# Full automation example
from scripts.runpod_api_client import RunPodAPIClient
import subprocess
import time

client = RunPodAPIClient()

# Create pod
pod = client.create_pod(...)
pod_id = pod['id']

# Wait for ready
while client.get_pod(pod_id)['desiredStatus'] != 'RUNNING':
    time.sleep(10)

# Upload files (runpodctl)
subprocess.run(["runpodctl", "send", ".", pod_id, "/workspace/tiny-icf"])

# Start training (runpodctl)
subprocess.run(["runpodctl", "exec", pod_id, "--", "bash", "start_training.sh"])

# Monitor
while True:
    pod_info = client.get_pod(pod_id)
    gpus = pod_info.get('runtime', {}).get('gpus', [])
    if gpus and gpus[0].get('gpuUtilPercent', 0) == 0:
        break
    time.sleep(60)

# Download results (runpodctl)
subprocess.run(["runpodctl", "receive", pod_id, "/workspace/tiny-icf/models", "models"])

# Stop pod
client.stop_pod(pod_id)
```

## Summary

The RunPod API is perfect for:
- ✅ Automated pod management
- ✅ Real-time monitoring
- ✅ Cost tracking
- ✅ Integration with CI/CD
- ✅ Programmatic workflows

Use `runpodctl` for:
- ✅ File upload/download
- ✅ Command execution
- ✅ SSH connections

**Best Practice**: Use API for lifecycle management, runpodctl for file/command operations.

