#!/bin/bash
# RunPod setup script - synced and executed remotely

set -e

cd /root/idf-est || { git clone https://github.com/arclabs561/tiny-icf.git /root/idf-est && cd /root/idf-est; }

export PATH="$HOME/.cargo/bin:$PATH"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Recreate venv if broken
if [ ! -f .venv/bin/python ] || [ ! -L .venv/bin/python ]; then
    echo "Recreating virtual environment..."
    rm -rf .venv
    uv venv
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install diffsort torch numpy pandas tqdm scipy lightning

echo "✓ Setup complete"
echo "Python: $(python3 --version)"
echo "UV: $(uv --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
