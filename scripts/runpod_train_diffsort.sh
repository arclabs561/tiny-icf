#!/bin/bash
# RunPod train diffsort script - synced and executed remotely

set -e

cd /root/idf-est
export PATH="$HOME/.cargo/bin:$PATH"

# Parse arguments
ARGS="${@}"

echo "Training with differentiable sorting..."
echo "Arguments: ${ARGS}"
echo ""

uv run scripts/train_diffsort.py \
    --data data/word_frequency.csv \
    --epochs 50 \
    --batch-size 64 \
    --method diffsort \
    --huber-weight 0.3 \
    --output models/model_diffsort.pt \
    --history training_history.json \
    ${ARGS} \
    2>&1 | tee train_diffsort.log

echo ""
echo "✓ Training complete"
echo "Model saved to: models/model_diffsort.pt"

