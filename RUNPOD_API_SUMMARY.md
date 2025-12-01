# RunPod API Direct Usage - Summary

## What We Learned

After researching the RunPod API documentation, we discovered:

### ✅ API CAN Do
- **Pod lifecycle management**: Create, start, stop, terminate pods
- **Pod information**: Get pod status, telemetry, runtime info
- **Resource queries**: List GPU types, check availability
- **Programmatic access**: Full GraphQL API for automation

### ❌ API CANNOT Do
- **Command execution**: No mutation to execute commands on pods
- **File transfer**: No API endpoints for upload/download
- **SSH management**: No API for SSH connections

## Solution Architecture

We've created a **hybrid approach** that combines the best of both:

1. **GraphQL API** (`scripts/runpod_api_client.py`):
   - Pod creation and management
   - Status monitoring
   - Resource queries

2. **runpodctl** (still needed):
   - File upload/download
   - Command execution
   - SSH connections

3. **Auto-start scripts** (workaround):
   - Upload scripts to pods
   - Trigger via `runpodctl exec`
   - More reliable than complex inline commands

## Files Created

### `scripts/runpod_api_client.py`
Full-featured Python client for RunPod GraphQL API:
- `create_pod()` - Create on-demand pods
- `get_pod()` - Get pod information
- `list_pods()` - List all pods
- `start_pod()` / `stop_pod()` - Pod lifecycle
- `get_gpu_types()` - Resource queries
- CLI interface for all operations

### `RUNPOD_API_GUIDE.md`
Comprehensive guide covering:
- What the API can/cannot do
- Authentication methods
- Usage examples
- Workarounds for command execution
- Best practices

### Updated `scripts/runpod_batch.py`
Now supports `--use-api` flag to use GraphQL API for pod creation:
```bash
# Use API for pod management
uv run scripts/runpod_batch.py --use-api --gpu-type "NVIDIA GeForce RTX 4080 SUPER"

# Still uses runpodctl for file transfer and command execution
```

## Usage Examples

### Create Pod via API
```python
from scripts.runpod_api_client import RunPodAPIClient

client = RunPodAPIClient()
pod = client.create_pod(
    name="training-pod",
    gpu_type_id="NVIDIA GeForce RTX 4080 SUPER",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_count=1,
    volume_in_gb=50,
)
```

### CLI Usage
```bash
# Create pod
uv run scripts/runpod_api_client.py create \
    --name "my-pod" \
    --gpu-type "NVIDIA GeForce RTX 4080 SUPER" \
    --image "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

# List pods
uv run scripts/runpod_api_client.py list

# Get pod info
uv run scripts/runpod_api_client.py get <pod-id>

# Start/stop pod
uv run scripts/runpod_api_client.py start <pod-id>
uv run scripts/runpod_api_client.py stop <pod-id>
```

### Complete Workflow
```bash
# 1. Create pod via API
uv run scripts/runpod_batch.py --use-api --upload-only

# 2. Upload files (uses runpodctl)
# Already done in step 1

# 3. Start training (uses runpodctl exec or auto-start script)
uv run scripts/runpod_batch.py --pod-id <pod-id> --background

# 4. Monitor status (uses runpodctl exec)
uv run scripts/runpod_batch.py --pod-id <pod-id> --status

# 5. Download results (uses runpodctl)
uv run scripts/runpod_batch.py --pod-id <pod-id> --download-only
```

## Key Insights

1. **No direct command execution**: The API doesn't support executing commands on pods. This is by design - RunPod expects you to use SSH or entrypoints.

2. **Best approach**: 
   - Use **API** for pod lifecycle (create, start, stop)
   - Use **runpodctl** for file transfer and command execution
   - Use **auto-start scripts** for reliable non-interactive execution

3. **Why this works**:
   - API provides reliable, programmatic pod management
   - runpodctl handles file/command operations that API can't
   - Auto-start scripts provide idempotent, reliable execution

## Next Steps

1. **Test the API client** with your actual API key
2. **Use `--use-api` flag** in `runpod_batch.py` for pod creation
3. **Monitor pods** via API instead of polling `runpodctl get pod`
4. **Consider RunPod Serverless** for fully automated batch jobs (no pod management needed)

## References

- **GraphQL API Spec**: https://graphql-spec.runpod.io/
- **API Documentation**: https://docs.runpod.io/api-reference
- **GraphQL Examples**: https://docs.runpod.io/sdks/graphql/manage-pods
- **Serverless Docs**: https://docs.runpod.io/serverless/overview

