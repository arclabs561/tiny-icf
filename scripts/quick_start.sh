#!/bin/bash
# Quick start script for tiny-icf training and evaluation

set -e

echo "=========================================="
echo "tiny-icf Quick Start"
echo "=========================================="
echo ""

# Check if data exists
if [ ! -f "data/word_frequency.csv" ]; then
    echo "⚠️  Data file not found: data/word_frequency.csv"
    echo "   Run: ./scripts/download_data.sh"
    exit 1
fi

# Check if model exists
if [ ! -f "models/model.pt" ]; then
    echo "📦 No trained model found. Starting training..."
    echo ""
    
    # Quick training (5 epochs for testing)
    uv run scripts/train_best_practices.py \
        --data data/word_frequency.csv \
        --epochs 5 \
        --scheduler adaptive \
        --early-stop \
        --output models/model.pt \
        --history training_history.json \
        --log training.log
    
    echo ""
    echo "✅ Training complete!"
else
    echo "✓ Found existing model: models/model.pt"
fi

echo ""
echo "🔍 Running evaluation..."
uv run scripts/comprehensive_eval.py \
    --model models/model.pt \
    --data data/word_frequency.csv \
    --output eval_results.json

echo ""
echo "📊 Quick predictions test..."
uv run scripts/quick_predict.py \
    --model models/model.pt \
    --words "the apple xylophone qzxbjk"

echo ""
echo "✅ Quick start complete!"
echo ""
echo "Next steps:"
echo "  - Train longer: uv run scripts/train_best_practices.py --epochs 100 --data data/word_frequency.csv"
echo "  - Compare configs: uv run scripts/compare_loss_configs.py"
echo "  - Benchmark: uv run scripts/benchmark_training.py --data data/word_frequency.csv"

