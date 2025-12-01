# RunPod API Direct Usage Guide

## Overview

RunPod provides a **GraphQL API** at `https://api.runpod.io/graphql` for programmatic pod management. This guide explains what the API can and cannot do, and how to use it effectively.

## What the API CAN Do

The RunPod GraphQL API supports:

### Pod Management
- ✅ **Create pods** (`podFindAndDeployOnDemand`, `podRentInterruptable`)
- ✅ **Start pods** (`podResume`, `podBidResume`)
- ✅ **Stop pods** (`podStop`)
- ✅ **Terminate pods** (`podTerminate`)
- ✅ **Get pod information** (`pod` query)
- ✅ **List pods** (`myself { pods }`)

### Resource Information
- ✅ **List GPU types** (`gpuTypes`)
- ✅ **Get GPU availability** (`gpuTypes { lowestPrice }`)
- ✅ **Get pod telemetry** (CPU, GPU, memory usage)

## What the API CANNOT Do

The RunPod GraphQL API does **NOT** support:

- ❌ **Direct command execution** on pods
- ❌ **File upload/download** (use `runpodctl send/receive` or S3-compatible API)
- ❌ **SSH connection management** (use `runpodctl ssh`)

## Authentication

The API uses API keys for authentication. You can provide the key in two ways:

1. **Authorization header** (recommended):
   ```python
   headers = {
       "Authorization": f"Bearer {api_key}",
       "Content-Type": "application/json",
   }
   ```

2. **Query parameter**:
   ```
   https://api.runpod.io/graphql?api_key=${YOUR_API_KEY}
   ```

Get your API key from: https://www.console.runpod.io/user/settings

## Using the API Client

We provide a Python client (`scripts/runpod_api_client.py`) that wraps the GraphQL API:

```python
from scripts.runpod_api_client import RunPodAPIClient

# Initialize client (reads API key from ~/.cursor/mcp.json)
client = RunPodAPIClient()

# Create a pod
pod = client.create_pod(
    name="my-training-pod",
    gpu_type_id="NVIDIA GeForce RTX 4080 SUPER",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_count=1,
    volume_in_gb=50,
    container_disk_in_gb=30,
    volume_mount_path="/workspace",
)

print(f"Pod created: {pod['id']}")

# Get pod status
pod_info = client.get_pod(pod['id'])
print(f"Status: {pod_info['desiredStatus']}")

# Start pod
client.start_pod(pod['id'])

# Stop pod
client.stop_pod(pod['id'])
```

## CLI Usage

```bash
# Create a pod
uv run scripts/runpod_api_client.py create \
    --name "training-pod" \
    --gpu-type "NVIDIA GeForce RTX 4080 SUPER" \
    --image "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
    --gpu-count 1 \
    --volume-gb 50

# List pods
uv run scripts/runpod_api_client.py list

# Get pod info
uv run scripts/runpod_api_client.py get <pod-id>

# Start pod
uv run scripts/runpod_api_client.py start <pod-id>

# Stop pod
uv run scripts/runpod_api_client.py stop <pod-id>

# List GPU types
uv run scripts/runpod_api_client.py gpu-types
```

## Command Execution Workaround

Since the API doesn't support direct command execution, use one of these approaches:

### Method 1: Upload Script + Entrypoint (Recommended)

1. **Create a startup script** that runs your training:
   ```bash
   #!/bin/bash
   cd /workspace/tiny-icf
   uv run scripts/train_batch.py --epochs 50 --batch-size 256
   ```

2. **Upload script** using `runpodctl send`:
   ```bash
   runpodctl send start_training.sh <pod-id> /workspace/tiny-icf/
   ```

3. **Use pod entrypoint** or **SSH** to trigger:
   ```bash
   # Via SSH
   runpodctl ssh connect <pod-id>
   bash /workspace/tiny-icf/start_training.sh
   
   # Or via runpodctl exec
   runpodctl exec <pod-id> -- bash /workspace/tiny-icf/start_training.sh
   ```

### Method 2: Docker Entrypoint

Configure the pod to run your script automatically on startup:

```python
pod = client.create_pod(
    name="auto-training",
    gpu_type_id="...",
    image_name="...",
    docker_args="bash /workspace/tiny-icf/start_training.sh",
    # ... other args
)
```

### Method 3: Use RunPod Serverless

For non-interactive batch jobs, consider **RunPod Serverless** instead of Pods:

- Automatic scaling
- Pay per request
- No pod management needed
- Built-in job queue

See: https://docs.runpod.io/serverless/overview

## Complete Workflow Example

```python
from scripts.runpod_api_client import RunPodAPIClient
import subprocess
import time

# 1. Create pod via API
client = RunPodAPIClient()
pod = client.create_pod(
    name="batch-training",
    gpu_type_id="NVIDIA GeForce RTX 4080 SUPER",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_count=1,
    volume_in_gb=50,
    env=[{"key": "PYTHONUNBUFFERED", "value": "1"}],
)
pod_id = pod['id']
print(f"Pod created: {pod_id}")

# 2. Wait for pod to be ready
print("Waiting for pod to be ready...")
for _ in range(30):  # Wait up to 5 minutes
    pod_info = client.get_pod(pod_id)
    if pod_info['desiredStatus'] == 'RUNNING':
        break
    time.sleep(10)
else:
    raise Exception("Pod failed to start")

# 3. Upload project files (use runpodctl)
subprocess.run(
    ["runpodctl", "send", ".", pod_id, "/workspace/tiny-icf"],
    check=True,
)

# 4. Upload and execute training script (use runpodctl)
subprocess.run(
    ["runpodctl", "send", "start_training.sh", pod_id, "/workspace/tiny-icf/"],
    check=True,
)

# 5. Start training (use runpodctl exec or SSH)
subprocess.run(
    ["runpodctl", "exec", pod_id, "--", "bash", "/workspace/tiny-icf/start_training.sh"],
    check=True,
)

# 6. Monitor pod status via API
while True:
    pod_info = client.get_pod(pod_id)
    runtime = pod_info.get('runtime', {})
    gpus = runtime.get('gpus', [])
    
    if gpus:
        gpu_util = gpus[0].get('gpuUtilPercent', 0)
        print(f"GPU utilization: {gpu_util}%")
    
    time.sleep(60)

# 7. Stop pod when done (via API)
client.stop_pod(pod_id)
```

## API vs runpodctl

| Feature | GraphQL API | runpodctl |
|---------|-------------|-----------|
| Pod creation | ✅ | ✅ |
| Pod start/stop | ✅ | ✅ |
| Command execution | ❌ | ✅ |
| File transfer | ❌ | ✅ |
| SSH connection | ❌ | ✅ |
| Status monitoring | ✅ | ✅ |
| Programmatic access | ✅ | ⚠️ (CLI only) |

**Recommendation**: Use the **GraphQL API** for pod lifecycle management, and **runpodctl** for file transfer and command execution.

## Error Handling

The API client raises `RunPodAPIError` for API-related errors:

```python
from scripts.runpod_api_client import RunPodAPIClient, RunPodAPIError

try:
    client = RunPodAPIClient()
    pod = client.create_pod(...)
except RunPodAPIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## GraphQL Query Examples

### Create On-Demand Pod

```graphql
mutation {
  podFindAndDeployOnDemand(
    input: {
      name: "training-pod"
      gpuTypeId: "NVIDIA GeForce RTX 4080 SUPER"
      imageName: "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
      gpuCount: 1
      volumeInGb: 50
      containerDiskInGb: 30
      minVcpuCount: 2
      minMemoryInGb: 15
      volumeMountPath: "/workspace"
      cloudType: ALL
      env: [{ key: "PYTHONUNBUFFERED", value: "1" }]
    }
  ) {
    id
    name
    machineId
    desiredStatus
  }
}
```

### Get Pod Status

```graphql
query {
  pod(input: { podId: "your-pod-id" }) {
    id
    name
    desiredStatus
    runtime {
      uptimeInSeconds
      container {
        cpuPercent
        memoryPercent
      }
      gpus {
        id
        gpuUtilPercent
        memoryUtilPercent
      }
    }
  }
}
```

### List All Pods

```graphql
query {
  myself {
    pods {
      id
      name
      desiredStatus
      costPerHr
    }
  }
}
```

## Best Practices

1. **Use API for automation**: Pod lifecycle management via API is more reliable than CLI
2. **Combine with runpodctl**: Use API for pod management, runpodctl for file/command operations
3. **Monitor pod status**: Poll `get_pod()` to check when pod is ready before uploading files
4. **Handle errors**: Always wrap API calls in try/except blocks
5. **Use persistent volumes**: Mount volumes at `/workspace` to persist data across pod restarts
6. **Set environment variables**: Use `env` parameter to configure pod behavior

## References

- **GraphQL API Spec**: https://graphql-spec.runpod.io/
- **API Documentation**: https://docs.runpod.io/api-reference
- **GraphQL Examples**: https://docs.runpod.io/sdks/graphql/manage-pods
- **API Keys**: https://www.console.runpod.io/user/settings

