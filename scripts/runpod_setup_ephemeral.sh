#!/bin/bash
# Setup script for ephemeral RunPod instances
# Handles disk space, installs dependencies, prepares environment

set -e

cd /root/idf-est

echo "=== Ephemeral Pod Setup ==="

# Setup PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Clean up disk space
echo "Cleaning disk space..."
rm -rf ~/.cache/uv/.tmp* 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true
df -h / | tail -1

# Install uv if needed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Create fresh venv
echo "Creating virtual environment..."
rm -rf .venv
uv venv --python python3.12

# Activate
source .venv/bin/activate

# Install core dependencies with pip (more reliable on ephemeral pods)
echo "Installing dependencies..."
pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch (CPU version to save space, or CUDA if available)
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA..."
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch CPU-only..."
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install --no-cache-dir numpy pandas tqdm scipy

# Install package (skip optional dependencies for now)
echo "Installing tiny-icf package (core only)..."
pip install --no-cache-dir -e . --no-deps
pip install --no-cache-dir lightning  # For non-interactive training if needed

# Verify
echo "Verifying installation..."
python -c "import torch, numpy, pandas, tiny_icf; print('✓ All imports work')"

echo ""
echo "=== Setup Complete ==="
echo "Disk space:"
df -h / | tail -1
echo ""
echo "Ready for training!"

