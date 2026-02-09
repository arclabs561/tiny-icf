# GPU Scaling Solution for Training

## Overview

Based on codebase review, the training system:
- Uses `train_flexible_opportunistic.py` for flexible training
- Detects GPUs opportunistically (currently 6x A40 GPUs available)
- Has checkpoint/resume capability
- Supports batch size scaling (64 base, up to 512 for aggressive configs)
- Currently has DataParallel disabled due to deadlock issues
- Uses mixed precision (AMP) for efficiency

## Scaling Solution

### 1. **Scale Up Script** (`scripts/scale_gpu_training.sh`)

**Features:**
- Launch AWS spot instances for additional GPU capacity
- Automatic cleanup after max-hours (safety)
- Tracks instances for easy management
- Integrates with existing training code

**Usage:**
```bash
# Launch GPU instance for 8 hours
./scripts/scale_gpu_training.sh up g4dn.xlarge 8

# Launch with S3 checkpoint sync
./scripts/scale_gpu_training.sh up g4dn.2xlarge 24 my-checkpoint-bucket

# Upload project to instance
./scripts/scale_gpu_training.sh upload              # Latest instance
./scripts/scale_gpu_training.sh upload i-1234567890 # Specific instance

# Check status
./scripts/scale_gpu_training.sh status

# Cleanup when done
./scripts/scale_gpu_training.sh down                # All instances
./scripts/scale_gpu_training.sh down i-1234567890    # Specific instance
```

**What it does:**
1. Launches spot instance with Deep Learning AMI (PyTorch pre-installed)
2. Sets up auto-shutdown after max-hours
3. Tracks instance ID for cleanup
4. Provides SSH connection details
5. User uploads project and runs training

### 2. **Safe Cleanup** (`scripts/cleanup_aws_resources.sh`)

**Features:**
- Dry-run mode (preview before cleanup)
- Finds all spot instances and requests
- Force mode for automatic cleanup
- Cost estimation

**Usage:**
```bash
# Preview what would be cleaned
./scripts/cleanup_aws_resources.sh --dry-run

# Actually cleanup
./scripts/cleanup_aws_resources.sh --force
```

## Integration with Existing Training

### Current Training Flow:
1. `train_flexible_opportunistic.py` detects available GPUs
2. Creates experiment configs (batch sizes: 64-512)
3. Scales batch size based on num_gpus
4. Saves checkpoints automatically
5. Supports resume from checkpoint

### Scaling Workflow:

**Option A: Scale on Current Pod**
- Current pod has 6x A40 GPUs
- Fix DataParallel deadlock → enable multi-GPU
- Scale batch size: 64 → 384 (64 × 6 GPUs)
- No additional cost, just better utilization

**Option B: Scale with AWS Spot**
- Launch additional spot instances when needed
- Transfer checkpoints via S3 (if configured)
- Resume training on new instance
- Auto-cleanup prevents cost overruns

## Safety Features

1. **Automatic Cleanup:**
   - Instance auto-shutdown after max-hours
   - Tracked instances can be cleaned with one command
   - Dry-run mode prevents accidental deletion

2. **Cost Control:**
   - Spot pricing (60-90% savings)
   - Max duration limits
   - Cost estimation in cleanup script

3. **Checkpoint Safety:**
   - Training code already saves checkpoints
   - Can resume on any instance
   - S3 sync (optional) for persistence

## Recommended Scaling Strategy

### For Current Setup (6x A40 Pod):
1. **Fix DataParallel deadlock** → enable 6-GPU training
2. **Scale batch size** → 64 → 384 effective batch
3. **No AWS needed** → current pod is sufficient

### For Additional Capacity:
1. **Use AWS Spot** for:
   - Longer training runs (>24h)
   - Multiple parallel experiments
   - When pod is unavailable

2. **Workflow:**
   ```bash
   # Launch with S3 checkpoint sync (recommended)
   ./scripts/scale_gpu_training.sh up g4dn.2xlarge 24 my-checkpoint-bucket
   
   # Upload project automatically
   ./scripts/scale_gpu_training.sh upload
   
   # SSH and start training (auto-resumes from checkpoint if exists)
   ssh -i ~/.ssh/tarek.pem ubuntu@<IP>
   cd ~/idf-est
   uv run scripts/train_flexible_opportunistic.py --data data/word_frequency.csv
   
   # Monitor status
   ./scripts/scale_gpu_training.sh status
   
   # Sync checkpoints manually (if needed)
   ./scripts/sync_checkpoints_s3.sh upload my-checkpoint-bucket
   
   # Cleanup when done
   ./scripts/scale_gpu_training.sh down
   ```

## New Features

1. **Automatic Project Upload** → `upload` command syncs project to instance
2. **S3 Checkpoint Sync** → Optional S3 bucket for checkpoint persistence
3. **Spot Interruption Handling** → Auto-checkpoint on 2-minute warning
4. **JSON Tracking** → Better instance tracking with metadata
5. **Selective Cleanup** → Cleanup specific instances or all

## Next Steps

1. **Fix DataParallel** → Enable 6-GPU training on current pod
2. **Test scaling script** → Launch test instance, verify cleanup
3. **Set up S3 bucket** → For checkpoint persistence across instances
4. **Monitor costs** → Track spot instance usage

## Cost Comparison

**Current Pod (6x A40):**
- Fixed cost (not spot)
- Already running
- No setup overhead

**AWS Spot (g4dn.xlarge):**
- ~$0.17/hour (68% savings vs on-demand)
- 1x T4 GPU, 4 vCPU, 16GB RAM
- Setup overhead: ~5-10 minutes

**AWS Spot (g4dn.2xlarge):**
- ~$0.24/hour
- 1x T4 GPU, 8 vCPU, 32GB RAM
- Better for larger batch sizes

**Recommendation:** Use current pod for most training, AWS Spot for overflow/long runs.

