#!/bin/bash
# Train NeuralNDCG for 100 epochs on RunPod
# This script is designed for ephemeral pods - sets up environment and trains

set -e

cd /root/idf-est

# Setup environment for ephemeral pod
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Create venv if needed
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate and install
source .venv/bin/activate
echo "Installing dependencies..."
uv pip install -e . 2>&1 | tail -5

# Ensure directories exist
mkdir -p logs models data

# Check data exists
if [ ! -f data/word_frequency_merged.csv ]; then
    echo "Warning: data/word_frequency_merged.csv not found"
    echo "Falling back to data/word_frequency.csv if available"
    DATA_FILE="${DATA_FILE:-data/word_frequency.csv}"
else
    DATA_FILE="data/word_frequency_merged.csv"
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: No data file found. Please download datasets first."
    exit 1
fi

echo "Using data file: $DATA_FILE"

# Start training with NeuralNDCG
echo "Starting 100-epoch NeuralNDCG training..."
echo "This will run in the background. Monitor with: tail -f logs/neural_ndcg_100ep.log"

export PYTHONUNBUFFERED=1

nohup python scripts/train_research_loss.py \
    --data "$DATA_FILE" \
    --epochs 100 \
    --batch-size 32 \
    --use-neural-ndcg \
    --output models/model_neural_ndcg_100ep.pt \
    --aim \
    --aim-experiment neural-ndcg-100ep \
    > logs/neural_ndcg_100ep.log 2>&1 &

PID=$!
echo $PID > logs/neural_ndcg_100ep.pid
echo "Training started with PID: $PID"
echo "Monitor progress: tail -f logs/neural_ndcg_100ep.log"
echo "Check GPU: nvidia-smi"

