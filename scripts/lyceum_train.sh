#!/bin/bash
# Automated Lyceum training workflow
# Usage: ./scripts/lyceum_train.sh [--hardware a100|cpu|h100] [--epochs 100] [--batch-size 256]

set -euo pipefail

# Defaults
HARDWARE_PROFILE="${1:-a100}"
EPOCHS="${2:-100}"
BATCH_SIZE="${3:-256}"
VM_NAME="tiny-icf-$(date +%Y%m%d-%H%M%S)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519.pub}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Lyceum Training Automation${NC}"
echo "Hardware: $HARDWARE_PROFILE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo ""

# Add uv tools to PATH
export PATH="$HOME/.local/bin:$PATH"

# Check prerequisites
if ! command -v lyceum &> /dev/null; then
    echo -e "${YELLOW}⚠️  Lyceum CLI not in PATH, trying to install...${NC}"
    uv tool install lyceum-cli || {
        echo -e "${RED}❌ Lyceum CLI not found${NC}"
        echo "Install with: uv tool install lyceum-cli"
        exit 1
    }
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
    echo "Generate with: ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519"
    exit 1
fi

# Check authentication
echo -e "${YELLOW}Checking authentication...${NC}"
if ! lyceum vms list-instances &> /dev/null 2>&1; then
    echo -e "${YELLOW}Not authenticated, attempting login...${NC}"
    lyceum auth login || {
        echo -e "${RED}❌ Authentication failed${NC}"
        exit 1
    }
fi
echo -e "${GREEN}✓ Authenticated${NC}"

# Start VM
echo -e "${YELLOW}Starting VM: $VM_NAME...${NC}"
set +e  # Don't exit on error, we'll handle it
VM_OUTPUT=$(lyceum vms start-instance \
    --hardware-profile "$HARDWARE_PROFILE" \
    --key "$(cat "$SSH_KEY")" \
    --name "$VM_NAME" 2>&1)
VM_EXIT_CODE=$?
set -e  # Re-enable exit on error

# Check for errors first
if [ $VM_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ Failed to start VM (exit code: $VM_EXIT_CODE)${NC}"
    echo "$VM_OUTPUT"
    if echo "$VM_OUTPUT" | grep -qi "503\|unavailable"; then
        echo ""
        echo -e "${YELLOW}⚠️  VM service is currently unavailable. Please try again later.${NC}"
        echo "Check status: lyceum vms list-instances"
    fi
    exit 1
fi

# Also check output for error messages even if exit code is 0
if echo "$VM_OUTPUT" | grep -qi "error\|503\|unavailable\|failed"; then
    echo -e "${RED}❌ Failed to start VM${NC}"
    echo "$VM_OUTPUT"
    if echo "$VM_OUTPUT" | grep -qi "503\|unavailable"; then
        echo ""
        echo -e "${YELLOW}⚠️  VM service is currently unavailable. Please try again later.${NC}"
        echo "Check status: lyceum vms list-instances"
    fi
    exit 1
fi

# Extract VM ID from output (format may vary)
VM_ID=$(echo "$VM_OUTPUT" | grep -oE '[a-z0-9-]{20,}' | head -1 || echo "")
if [ -z "$VM_ID" ]; then
    # Try alternative parsing - look for any UUID-like pattern
    VM_ID=$(echo "$VM_OUTPUT" | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1 || echo "")
fi

if [ -z "$VM_ID" ]; then
    echo -e "${RED}❌ Failed to extract VM ID from output${NC}"
    echo "Output was:"
    echo "$VM_OUTPUT"
    exit 1
fi

echo -e "${GREEN}✓ VM started: $VM_ID${NC}"

# Wait for VM to be ready
echo -e "${YELLOW}Waiting for VM to be ready (this may take a few minutes)...${NC}"
MAX_WAIT=600  # 10 minutes
ELAPSED=0
INTERVAL=30

while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS_OUTPUT=$(lyceum vms instance-status "$VM_ID" 2>/dev/null || echo "")
    VM_STATUS=$(echo "$STATUS_OUTPUT" | grep -i "status" | awk '{print $2}' | tr '[:upper:]' '[:lower:]' || echo "pending")
    
    if [ "$VM_STATUS" = "ready" ] || [ "$VM_STATUS" = "running" ] || [ "$VM_STATUS" = "active" ]; then
        echo -e "${GREEN}✓ VM is ready!${NC}"
        break
    fi
    
    echo "  Status: $VM_STATUS (waiting ${INTERVAL}s...)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ "$VM_STATUS" != "ready" ] && [ "$VM_STATUS" != "running" ] && [ "$VM_STATUS" != "active" ]; then
    echo -e "${RED}❌ VM did not become ready in time${NC}"
    echo "Check status: lyceum vms instance-status $VM_ID"
    exit 1
fi

# Get VM IP from status
STATUS_OUTPUT=$(lyceum vms instance-status "$VM_ID" 2>/dev/null || echo "")
VM_IP=$(echo "$STATUS_OUTPUT" | grep -i "ip\|address" | grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -1 || echo "")
if [ -z "$VM_IP" ]; then
    echo -e "${RED}❌ Could not get VM IP${NC}"
    exit 1
fi

echo -e "${GREEN}✓ VM IP: $VM_IP${NC}"

# Wait a bit more for SSH to be ready
echo -e "${YELLOW}Waiting for SSH...${NC}"
sleep 10

# Test SSH connection
SSH_KEY_PRIVATE="${SSH_KEY%.pub}"
MAX_SSH_ATTEMPTS=10
SSH_ATTEMPT=0

while [ $SSH_ATTEMPT -lt $MAX_SSH_ATTEMPTS ]; do
    if ssh -i "$SSH_KEY_PRIVATE" -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@"$VM_IP" "echo 'SSH ready'" &> /dev/null; then
        echo -e "${GREEN}✓ SSH connection ready${NC}"
        break
    fi
    SSH_ATTEMPT=$((SSH_ATTEMPT + 1))
    echo "  Attempt $SSH_ATTEMPT/$MAX_SSH_ATTEMPTS..."
    sleep 5
done

if [ $SSH_ATTEMPT -eq $MAX_SSH_ATTEMPTS ]; then
    echo -e "${RED}❌ SSH connection failed${NC}"
    echo "Try manually: ssh -i $SSH_KEY_PRIVATE ubuntu@$VM_IP"
    exit 1
fi

# Setup VM
echo -e "${YELLOW}Setting up VM...${NC}"
ssh -i "$SSH_KEY_PRIVATE" -o StrictHostKeyChecking=no ubuntu@"$VM_IP" << 'ENDSSH'
    set -e
    # Update system
    sudo apt-get update -qq
    sudo apt-get install -y -qq git python3-pip python3-venv curl
    
    # Install uv if not present
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # Clone or update repo (assuming it's a git repo)
    if [ -d "idf-est" ]; then
        cd idf-est
        git pull
    else
        echo "⚠️  Repository not found. Please clone manually or upload files."
    fi
ENDSSH

# Upload project files
echo -e "${YELLOW}Uploading project files...${NC}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create tarball of essential files
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

cd "$PROJECT_ROOT"
tar -czf "$TEMP_DIR/project.tar.gz" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='models/*.pt' \
    --exclude='models/*.ckpt' \
    --exclude='*.log' \
    --exclude='node_modules' \
    src/ scripts/ pyproject.toml uv.lock data/ tests/ 2>/dev/null || true

scp -i "$SSH_KEY_PRIVATE" -o StrictHostKeyChecking=no "$TEMP_DIR/project.tar.gz" ubuntu@"$VM_IP":~/project.tar.gz

# Extract on VM
ssh -i "$SSH_KEY_PRIVATE" -o StrictHostKeyChecking=no ubuntu@"$VM_IP" << ENDSSH
    set -e
    cd ~
    if [ -d "idf-est" ]; then
        cd idf-est
        tar -xzf ~/project.tar.gz
    else
        mkdir -p idf-est
        cd idf-est
        tar -xzf ~/project.tar.gz
    fi
    rm ~/project.tar.gz
ENDSSH

# Install dependencies and run training
echo -e "${YELLOW}Installing dependencies and starting training...${NC}"
ssh -i "$SSH_KEY_PRIVATE" -o StrictHostKeyChecking=no ubuntu@"$VM_IP" << ENDSSH
    set -e
    cd ~/idf-est
    
    # Install dependencies
    uv sync
    
    # Verify GPU
    echo "GPU Status:"
    nvidia-smi || echo "No GPU detected (using CPU)"
    
    # Run training
    echo "Starting training..."
    uv run python -m tiny_icf.train_lightning \
        --data data/word_frequency.csv \
        --output-dir models/lyceum \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr 2e-3 \
        --precision 16-mixed \
        --devices 1 \
        2>&1 | tee training.log
    
    echo "Training complete!"
ENDSSH

# Download results
echo -e "${YELLOW}Downloading results...${NC}"
mkdir -p "$PROJECT_ROOT/models/lyceum"
scp -i "$SSH_KEY_PRIVATE" -o StrictHostKeyChecking=no \
    ubuntu@"$VM_IP":~/idf-est/models/lyceum/*.pt \
    "$PROJECT_ROOT/models/lyceum/" 2>/dev/null || echo "No .pt files found"

scp -i "$SSH_KEY_PRIVATE" -o StrictHostKeyChecking=no \
    ubuntu@"$VM_IP":~/idf-est/models/lyceum/*.ckpt \
    "$PROJECT_ROOT/models/lyceum/" 2>/dev/null || echo "No .ckpt files found"

scp -i "$SSH_KEY_PRIVATE" -o StrictHostKeyChecking=no \
    ubuntu@"$VM_IP":~/idf-est/training.log \
    "$PROJECT_ROOT/training_lyceum.log" 2>/dev/null || echo "No log file found"

# Ask about termination
echo ""
echo -e "${GREEN}✅ Training complete!${NC}"
echo "Results downloaded to: $PROJECT_ROOT/models/lyceum/"
echo ""
read -p "Terminate VM $VM_ID? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    lyceum vms terminate "$VM_ID" -f
    echo -e "${GREEN}✓ VM terminated${NC}"
else
    echo "VM still running: $VM_ID"
    echo "SSH: ssh -i $SSH_KEY_PRIVATE ubuntu@$VM_IP"
    echo "Terminate later: lyceum vms terminate-instance $VM_ID"
fi

echo ""
echo -e "${GREEN}🎉 All done!${NC}"

