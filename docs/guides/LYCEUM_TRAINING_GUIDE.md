# Training with Lyceum VMs

Guide for training the tiny-icf model on Lyceum cloud VMs.

## Prerequisites

1. **Install Lyceum CLI**
   ```bash
   # Install via your package manager or download from:
   # https://docs.lyceum.technology
   ```

2. **Authenticate**
   ```bash
   lyceum auth login
   ```

3. **Prepare SSH Key**
   ```bash
   # Generate if you don't have one
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
   
   # Or use existing key
   cat ~/.ssh/id_ed25519.pub  # or id_rsa.pub
   ```

## Quick Start

### 1. Check Hardware Availability

```bash
lyceum vms availability
```

### 2. Start a VM

```bash
# A100 instance (recommended for training)
lyceum vms start \
  -h a100 \
  -k "$(cat ~/.ssh/id_ed25519.pub)" \
  -n "tiny-icf-training"

# Wait for "VM ready" message with SSH command
```

### 3. SSH into VM

```bash
# Use the SSH command provided by lyceum
ssh -i ~/.ssh/id_ed25519 ubuntu@<ip-address>
```

### 4. Setup on VM

```bash
# Clone your repository
git clone <your-repo-url>
cd idf-est  # or your repo name

# Install dependencies
uv sync  # or pip install -r requirements.txt

# Verify GPU
nvidia-smi
```

### 5. Run Training

```bash
# Lightning training (recommended)
uv run python -m tiny_icf.train_lightning \
  --data data/word_frequency.csv \
  --output-dir models/lyceum \
  --epochs 100 \
  --batch-size 256 \
  --lr 2e-3 \
  --devices 1 \
  --precision 16-mixed

# Or standard training
uv run python -m tiny_icf.train \
  --data data/word_frequency.csv \
  --epochs 100 \
  --output models/model.pt
```

### 6. Download Results

```bash
# From your local machine, use scp to download models
scp -i ~/.ssh/id_ed25519 \
  ubuntu@<ip>:/home/ubuntu/idf-est/models/*.pt \
  ./models/
```

### 7. Terminate VM

```bash
# List VMs to get ID
lyceum vms list

# Terminate
lyceum vms terminate <vm-id> -f
```

## Hardware Profiles

### CPU Profile
- **Use case**: Development, testing, small datasets
- **Command**: `lyceum vms start -h cpu ...`
- **Specs**: 4 vCPU, 16 GB RAM, no GPU

### A100 Profile (Recommended)
- **Use case**: Production training, large datasets
- **Command**: `lyceum vms start -h a100 ...`
- **Specs**: 8 vCPU, 64 GB RAM, 1x NVIDIA A100
- **Cost**: ~$2-3/hour

### H100 Profile
- **Use case**: Maximum performance, very large models
- **Command**: `lyceum vms start -h h100 ...`
- **Specs**: 8 vCPU, 80 GB RAM, 1x NVIDIA H100
- **Cost**: Higher than A100

## Training Scripts

### Lightning Training (Recommended)

Best for non-interactive batch jobs with automatic checkpointing:

```bash
uv run python -m tiny_icf.train_lightning \
  --data data/word_frequency.csv \
  --output-dir models/lyceum \
  --epochs 100 \
  --batch-size 256 \
  --lr 2e-3 \
  --max-length 20 \
  --augment-prob 0.2 \
  --curriculum-stages 5 \
  --warmup-epochs 5 \
  --early-stopping-patience 10 \
  --devices 1 \
  --precision 16-mixed
```

**Features:**
- Automatic mixed precision (16-bit)
- Model checkpointing (top 3 + last)
- Early stopping
- CSV logging
- Curriculum learning
- GPU acceleration

### Standard Training

For interactive debugging or simpler workflows:

```bash
uv run python -m tiny_icf.train \
  --data data/word_frequency.csv \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-3 \
  --output models/model.pt
```

## Monitoring Training

### On VM

```bash
# Watch training logs
tail -f training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h
```

### From Local Machine

```bash
# SSH with port forwarding for TensorBoard (if using)
ssh -i ~/.ssh/id_ed25519 -L 6006:localhost:6006 ubuntu@<ip>

# Then on VM:
tensorboard --logdir models/lyceum/logs --port 6006
# Access at http://localhost:6006 locally
```

## Data Transfer

### Upload Data to VM

```bash
# From local machine
scp -i ~/.ssh/id_ed25519 \
  data/word_frequency.csv \
  ubuntu@<ip>:/home/ubuntu/idf-est/data/
```

### Download Models from VM

```bash
# Download all checkpoints
scp -i ~/.ssh/id_ed25519 \
  ubuntu@<ip>:/home/ubuntu/idf-est/models/lyceum/*.ckpt \
  ./models/lyceum/

# Download final model
scp -i ~/.ssh/id_ed25519 \
  ubuntu@<ip>:/home/ubuntu/idf-est/models/lyceum/model_final.pt \
  ./models/
```

## Cost Optimization

1. **Check availability first**: `lyceum vms availability`
2. **Use appropriate hardware**: CPU for dev, A100 for training
3. **Monitor usage**: `lyceum vms status <vm-id>` shows billing
4. **Terminate promptly**: Don't leave VMs running idle
5. **Batch jobs**: Use Lightning training for non-interactive runs

## Troubleshooting

### VM Not Starting

```bash
# Check availability
lyceum vms availability

# Try different hardware profile
lyceum vms start -h cpu ...  # Fallback to CPU
```

### SSH Connection Issues

```bash
# Verify key format
cat ~/.ssh/id_ed25519.pub

# Check VM status
lyceum vms status <vm-id>

# Try with verbose SSH
ssh -v -i ~/.ssh/id_ed25519 ubuntu@<ip>
```

### Training Errors

```bash
# Check GPU availability
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check disk space
df -h

# Verify data file
head data/word_frequency.csv
```

## Example Workflow

```bash
# 1. Check availability
lyceum vms availability

# 2. Start VM
VM_ID=$(lyceum vms start -h a100 \
  -k "$(cat ~/.ssh/id_ed25519.pub)" \
  -n "training-$(date +%Y%m%d)" \
  | grep -o 'vm-[a-z0-9]*' | head -1)

# 3. Wait and get IP
sleep 60
VM_IP=$(lyceum vms status $VM_ID | grep "IP Address" | awk '{print $3}')

# 4. Upload data
scp -i ~/.ssh/id_ed25519 data/word_frequency.csv ubuntu@$VM_IP:~/idf-est/data/

# 5. SSH and train
ssh -i ~/.ssh/id_ed25519 ubuntu@$VM_IP << 'EOF'
cd idf-est
uv sync
uv run python -m tiny_icf.train_lightning \
  --data data/word_frequency.csv \
  --output-dir models/lyceum \
  --epochs 100 \
  --batch-size 256
EOF

# 6. Download results
scp -i ~/.ssh/id_ed25519 ubuntu@$VM_IP:~/idf-est/models/lyceum/*.pt ./models/

# 7. Terminate
lyceum vms terminate $VM_ID -f
```

## Best Practices

1. **Always check availability** before starting expensive hardware
2. **Use Lightning training** for production runs (better checkpointing)
3. **Monitor costs** with `lyceum vms status`
4. **Download results immediately** after training completes
5. **Terminate VMs** as soon as you're done
6. **Use SSH keys** (not passwords) for security
7. **Keep training scripts** in version control for reproducibility

