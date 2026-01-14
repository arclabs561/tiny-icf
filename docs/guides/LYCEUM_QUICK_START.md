# Lyceum Training - Quick Start

## One-Command Training (Fully Automated)

```bash
./scripts/lyceum_train.sh a100 100 256
```

This script will:
1. ✅ Check authentication
2. ✅ Check hardware availability  
3. ✅ Start an A100 VM
4. ✅ Wait for it to be ready
5. ✅ Upload your project files
6. ✅ Install dependencies
7. ✅ Run training with PyTorch Lightning
8. ✅ Download results
9. ✅ Ask if you want to terminate the VM

**Prerequisites:**
- Lyceum CLI installed
- Authenticated: `lyceum auth login`
- SSH key: `~/.ssh/id_ed25519.pub` (or set `SSH_KEY` env var)

## Manual Steps (If You Prefer)

### 1. Authenticate
```bash
lyceum auth login
```

### 2. Start VM
```bash
lyceum vms start -h a100 -k "$(cat ~/.ssh/id_ed25519.pub)" -n "training"
```

### 3. Wait for Ready, Then SSH
```bash
# Get IP from: lyceum vms status <vm-id>
ssh -i ~/.ssh/id_ed25519 ubuntu@<ip>
```

### 4. On VM: Setup & Train
```bash
cd ~/idf-est  # or wherever you clone/upload the project
uv sync
uv run python -m tiny_icf.train_lightning \
  --data data/word_frequency.csv \
  --output-dir models/lyceum \
  --epochs 100 \
  --batch-size 256 \
  --precision 16-mixed
```

### 5. Download Results (from local machine)
```bash
./scripts/lyceum_download_results.sh <vm-ip>
```

### 6. Terminate VM
```bash
lyceum vms terminate <vm-id> -f
```

## Helper Scripts

- `scripts/lyceum_train.sh` - Full automation
- `scripts/lyceum_setup_vm.sh` - VM setup script
- `scripts/lyceum_upload_data.sh` - Upload data files
- `scripts/lyceum_download_results.sh` - Download results
- `scripts/lyceum_quick_start.sh` - Shows manual steps

## Troubleshooting

**VM not starting?**
```bash
lyceum vms availability  # Check what's available
```

**SSH connection issues?**
```bash
# Check VM status
lyceum vms status <vm-id>

# Try verbose SSH
ssh -v -i ~/.ssh/id_ed25519 ubuntu@<ip>
```

**Training errors?**
- Check GPU: `nvidia-smi` (on VM)
- Check data: `ls -lh data/word_frequency.csv` (on VM)
- Check logs: `tail -f training.log` (on VM)

## Cost Optimization

- Use `cpu` profile for testing: `./scripts/lyceum_train.sh cpu 10 64`
- Monitor costs: `lyceum vms status <vm-id>`
- Terminate promptly when done

## Full Documentation

See [LYCEUM_TRAINING_GUIDE.md](LYCEUM_TRAINING_GUIDE.md) for complete details.

