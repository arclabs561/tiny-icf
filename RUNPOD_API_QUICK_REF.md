# RunPod API - Quick Reference Card

## Essential Commands

### Pod Management
```bash
# List all pods
uv run scripts/runpod_api_client.py list

# Get pod details
uv run scripts/runpod_api_client.py get <pod-id>

# Create pod
uv run scripts/runpod_api_client.py create \
    --name "my-pod" \
    --gpu-type "NVIDIA GeForce RTX 4080 SUPER" \
    --image "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

# Start pod
uv run scripts/runpod_api_client.py start <pod-id>

# Stop pod
uv run scripts/runpod_api_client.py stop <pod-id>

# Terminate pod (permanent)
uv run scripts/runpod_api_client.py terminate <pod-id>
```

### GPU Queries
```bash
# List all GPU types
uv run scripts/runpod_api_client.py gpu-types

# Find specific GPU
uv run scripts/runpod_api_client.py gpu-types --id "NVIDIA GeForce RTX 4080"
```

### Monitoring
```bash
# Real-time monitoring
uv run scripts/runpod_monitor.py <pod-id> --interval 10 --duration 300

# One-time status
uv run scripts/runpod_batch.py --pod-id <pod-id> --status
```

### Batch Training (with API)
```bash
# Create pod and upload (API)
uv run scripts/runpod_batch.py --use-api --upload-only

# Start training (runpodctl)
uv run scripts/runpod_batch.py --pod-id <pod-id> --background

# Check status
uv run scripts/runpod_batch.py --pod-id <pod-id> --status

# Download results (runpodctl)
uv run scripts/runpod_batch.py --pod-id <pod-id> --download-only
```

## Python API Usage

```python
from scripts.runpod_api_client import RunPodAPIClient

# Initialize
client = RunPodAPIClient()

# Create pod
pod = client.create_pod(
    name="training-pod",
    gpu_type_id="NVIDIA GeForce RTX 4080 SUPER",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_count=1,
    volume_in_gb=50,
)
pod_id = pod['id']

# Get pod info
pod_info = client.get_pod(pod_id)
print(f"Status: {pod_info['desiredStatus']}")
print(f"GPU Util: {pod_info['runtime']['gpus'][0]['gpuUtilPercent']}%")

# List pods
pods = client.list_pods()

# Start/stop
client.start_pod(pod_id)
client.stop_pod(pod_id)
```

## What API Can Do ✅

- Create/start/stop/terminate pods
- Get pod information and telemetry
- List all pods
- Query GPU types and availability
- Monitor CPU/GPU/memory usage
- Track costs and uptime

## What API Cannot Do ❌

- Execute commands (use `runpodctl exec`)
- Upload/download files (use `runpodctl send/receive`)
- SSH connections (use `runpodctl ssh`)

## Best Practice

**Use API for**: Pod lifecycle, monitoring, queries
**Use runpodctl for**: Files, commands, SSH

## Current Pod Status

```bash
# Quick status check
uv run scripts/runpod_api_client.py get <pod-id> | jq '{status: .desiredStatus, cost: .costPerHr, gpu: .runtime.gpus[0].gpuUtilPercent}'
```

## Cost Tracking

```bash
# Current hourly cost
uv run scripts/runpod_api_client.py get <pod-id> | jq '.costPerHr'

# Estimated total cost
pod=$(uv run scripts/runpod_api_client.py get <pod-id>)
cost=$(echo "$pod" | jq '.costPerHr')
uptime=$(echo "$pod" | jq '.runtime.uptimeInSeconds')
echo "scale=4; $cost * ($uptime / 3600)" | bc
```

