#!/bin/bash
# Quick training test for RunPod - runs 5 epochs to verify everything works

set -e

cd /root/idf-est
export PATH="$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1

# Create logs directory
mkdir -p logs

echo "=== Quick Training Test ==="
echo "Date: $(date)"
echo ""

# Parse arguments
AIM_ARGS=""
if [ "$1" == "--aim" ] || [ "$1" == "-a" ]; then
    AIM_ARGS="--aim --aim-experiment quick-test"
    echo "Aim tracking enabled"
fi

# Run quick training (5 epochs, small batch)
echo "Starting quick training test..."
echo "Epochs: 5"
echo "Batch size: 32"
echo ""

python3 -u scripts/train_best_practices.py \
    --data data/word_frequency.csv \
    --epochs 5 \
    --batch-size 32 \
    --output models/model_quick_test.pt \
    --early-stop \
    --early-stop-patience 3 \
    ${AIM_ARGS} \
    > logs/quick_train.log 2>&1

echo ""
echo "=== Quick Test Complete ==="
echo "Check results:"
echo "  - Model: models/model_quick_test.pt"
echo "  - Log: logs/quick_train.log"
if [ -n "$AIM_ARGS" ]; then
    echo "  - Aim: experiment 'quick-test'"
fi

