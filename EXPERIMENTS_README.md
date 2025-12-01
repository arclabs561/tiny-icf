# RunPod API Experiments

This document describes the experiments to test the RunPod API integration.

## Quick Start

Run all experiments in dry-run mode (safe, no pods created):
```bash
uv run scripts/runpod_experiments.py --dry-run
```

Run all experiments and create a test pod:
```bash
uv run scripts/runpod_experiments.py --create-pod
```

Run specific experiments:
```bash
# Test only API client and GPU queries
uv run scripts/runpod_experiments.py --dry-run --skip 3 4 5 6 7 8

# Test with existing pod
uv run scripts/runpod_experiments.py --pod-id <your-pod-id> --skip 5
```

## Experiments

### Experiment 1: API Client Initialization
- Tests API key retrieval from `~/.cursor/mcp.json`
- Tests API client initialization
- Validates API endpoint connectivity

**Expected**: ✅ API client initializes successfully

### Experiment 2: Query GPU Types
- Lists all available GPU types
- Finds target GPU (RTX 4080 SUPER)
- Returns GPU type ID for pod creation

**Expected**: ✅ GPU types retrieved, target GPU found

### Experiment 3: List Existing Pods
- Lists all user pods via API
- Identifies running pods
- Returns pod ID for subsequent experiments

**Expected**: ✅ Pods listed (may be empty)

### Experiment 4: Get Pod Information
- Retrieves detailed pod information
- Shows status, runtime, telemetry
- Validates pod query functionality

**Expected**: ✅ Pod info retrieved with runtime data

### Experiment 5: Create Test Pod
- Creates a test pod via GraphQL API
- Configures with standard training image
- Sets up persistent volume

**Expected**: ✅ Pod created successfully (if not dry-run)

### Experiment 6: runpodctl Integration
- Tests `runpodctl get pod`
- Tests `runpodctl exec` command execution
- Validates hybrid API + runpodctl workflow

**Expected**: ✅ runpodctl operations work on API-created pod

### Experiment 7: Batch Script Integration
- Tests `--use-api` flag in batch script
- Validates batch script can use API client
- Tests status checking

**Expected**: ✅ Batch script supports API integration

### Experiment 8: Error Handling
- Tests invalid pod ID handling
- Tests invalid GPU type handling
- Validates error messages

**Expected**: ✅ Errors caught and reported correctly

## Example Output

```
🧪 RunPod API Experiments
============================================================
Dry run: True
Skip experiments: set()

============================================================
EXPERIMENT 1: API Client Initialization
============================================================
📝 Testing API client initialization...
✅ API client initialized successfully
   API URL: https://api.runpod.io/graphql

============================================================
EXPERIMENT 2: Query GPU Types
============================================================
📝 Querying GPU types...
✅ Found 45 GPU types

   Sample GPU types:
   - NVIDIA GeForce RTX 3090: NVIDIA GeForce RTX 3090 (24GB)
   - NVIDIA GeForce RTX 4080 SUPER: NVIDIA GeForce RTX 4080 SUPER (16GB)
   ...

✅ Found target GPU: NVIDIA GeForce RTX 4080 SUPER
...
```

## Cleanup

If you created a test pod, remember to clean it up:

```bash
# Stop the pod
uv run scripts/runpod_api_client.py stop <pod-id>

# Or terminate it (permanent deletion)
uv run scripts/runpod_api_client.py terminate <pod-id>
```

## Troubleshooting

### API Key Not Found
```
❌ API Error: No API key found
```
**Solution**: Ensure `RUNPOD_API_KEY` is set in `~/.cursor/mcp.json`

### runpodctl Not Found
```
❌ runpodctl not found in PATH
```
**Solution**: Install runpodctl: https://docs.runpod.io/runpodctl/overview

### Pod Creation Fails
```
❌ Error creating pod: GraphQL errors: ...
```
**Solution**: Check GPU availability, API key permissions, or try different GPU type

## Next Steps After Experiments

1. **If all experiments pass**: You're ready to use the API integration!
2. **If pod creation works**: Try the full training workflow with `--use-api`
3. **If errors occur**: Check the error messages and API documentation

## Integration Testing

After running experiments, test the full workflow:

```bash
# 1. Create pod via API
uv run scripts/runpod_batch.py --use-api --upload-only

# 2. Start training
uv run scripts/runpod_batch.py --pod-id <pod-id> --background

# 3. Monitor status
uv run scripts/runpod_batch.py --pod-id <pod-id> --status

# 4. Download results
uv run scripts/runpod_batch.py --pod-id <pod-id> --download-only
```

