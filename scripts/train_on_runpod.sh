#!/bin/bash
# Training script for RunPod GPU instance
# This script sets up the environment and runs training on a RunPod GPU

set -e

echo "🚀 RunPod Training Setup"
echo "========================"

# Check for required files
if [ ! -f "data/word_frequency.csv" ]; then
    echo "❌ Error: data/word_frequency.csv not found"
    exit 1
fi

# Training configuration
DATA_FILE="${DATA_FILE:-data/word_frequency.csv}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
OUTPUT="${OUTPUT:-models/model_runpod.pt}"
DEVICE="${DEVICE:-auto}"

echo "Configuration:"
echo "  Data: $DATA_FILE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Output: $OUTPUT"
echo "  Device: $DEVICE"
echo ""

# Run training (use curriculum training for better results)
echo "Starting training..."
python -m tiny_icf.train_curriculum \
    --data "${DATA_FILE:-data/multilingual/multilingual_combined.csv}" \
    --typo-corpus "data/typos/github_typos.csv" \
    --multilingual \
    --epochs "${EPOCHS:-50}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --lr "${LR:-1e-3}" \
    --curriculum-stages 5 \
    --output "$OUTPUT" \
    --device "${DEVICE:-auto}"

echo ""
echo "✅ Training complete!"
echo "Model saved to: $OUTPUT"

