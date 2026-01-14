#!/bin/bash
# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
"""
Flexible training setup script - adapts to available resources.

Opportunistically sets up training based on what's available:
- GPU vs CPU
- Available memory
- Python version
- Existing installations
"""

set -e

SSH_HOST="${1:-root@194.68.245.50}"
SSH_PORT="${2:-22106}"
SSH_KEY="${3:-~/.ssh/id_ed25519}"

echo "=========================================="
echo "Flexible Training Setup"
echo "=========================================="
echo "Host: $SSH_HOST"
echo "Port: $SSH_PORT"
echo ""

# Test connection
echo "Testing SSH connection..."
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "echo '✓ Connection successful'" || {
    echo "❌ Connection failed"
    exit 1
}

# Detect environment
echo ""
echo "Detecting environment..."
ENV_INFO=$(ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" << 'EOF'
{
    echo "=== System Info ==="
    uname -a
    echo ""
    echo "=== Python ==="
    python3 --version 2>/dev/null || echo "Python3 not found"
    python3 -m pip --version 2>/dev/null || echo "pip not found"
    echo ""
    echo "=== GPU ==="
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
    echo ""
    echo "=== Memory ==="
    free -h | head -2
    echo ""
    echo "=== Disk ==="
    df -h / | tail -1
    echo ""
    echo "=== UV ==="
    uv --version 2>/dev/null || echo "uv not installed"
} | cat
EOF
)

echo "$ENV_INFO"

# Check if project exists
echo ""
echo "Checking project directory..."
PROJECT_EXISTS=$(ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "test -d /root/idf-est && echo 'yes' || echo 'no'")

if [ "$PROJECT_EXISTS" = "no" ]; then
    echo "Project not found. Setting up..."
    
    # Create directory
    ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "mkdir -p /root/idf-est"
    
    # Transfer essential files
    echo "Transferring project files..."
    scp -P "$SSH_PORT" -i "$SSH_KEY" -r \
        src/ \
        scripts/train_*.py \
        scripts/setup_*.sh \
        data/word_frequency.csv \
        "$SSH_HOST:/root/idf-est/" 2>/dev/null || echo "Some files may not exist, continuing..."
else
    echo "✓ Project directory exists"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" << 'EOF'
cd /root/idf-est

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install Python dependencies using uv
echo "Installing Python dependencies..."
if [ -f "pyproject.toml" ]; then
    uv pip install --system -e .
else
    # Install essential packages
    uv pip install --system torch numpy pandas tqdm scipy
fi

echo "✓ Dependencies installed"
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run training with: ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST 'cd /root/idf-est && ...'"
echo "2. Monitor with: ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST 'cd /root/idf-est && ...'"

