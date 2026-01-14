#!/bin/bash
# Setup script to run on Lyceum VM
# This prepares the environment for training

set -euo pipefail

echo "🔧 Setting up training environment..."

# Update system
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git \
    python3-pip \
    python3-venv \
    curl \
    build-essential \
    htop \
    tmux

# Install uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# Verify GPU
echo ""
echo "GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "No GPU detected (will use CPU)"
fi

# Check Python
echo ""
echo "Python version:"
python3 --version

# Check uv
echo ""
echo "uv version:"
uv --version

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. cd ~/idf-est"
echo "  2. uv sync"
echo "  3. uv run python -m tiny_icf.train_lightning --data data/word_frequency.csv --output-dir models/lyceum --epochs 100"

