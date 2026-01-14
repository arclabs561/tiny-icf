#!/bin/bash
# Progressive training: Start simple, increase complexity
# Runs experiments in order of increasing difficulty

set -e

DATA_FILE="${1:-data/word_frequency.csv}"
OUTPUT_DIR="${2:-models/progressive}"

echo "🎯 Progressive Training: Increasing Complexity"
echo "=============================================="
echo ""

# Stage 1: Simple baseline (1 experiment)
echo "📊 Stage 1: Baseline (Simple)"
echo "   Experiment: standard_enhanced"
echo "   Batch: 64, Epochs: 50"
uv run scripts/train_flexible_opportunistic.py \
    --data "$DATA_FILE" \
    --max_experiments 1 \
    --experiments standard_enhanced \
    --train_split 0.8 \
    2>&1 | tee "${OUTPUT_DIR}/stage1_baseline.log"

echo ""
echo "✅ Stage 1 complete"
echo ""

# Stage 2: Add complexity (2 experiments)
echo "📊 Stage 2: Moderate Complexity"
echo "   Experiments: standard_enhanced, residual_listwise"
echo "   Adds: Residual model + LambdaRank"
uv run scripts/train_flexible_opportunistic.py \
    --data "$DATA_FILE" \
    --max_experiments 2 \
    --experiments standard_enhanced residual_listwise \
    --train_split 0.8 \
    2>&1 | tee "${OUTPUT_DIR}/stage2_moderate.log"

echo ""
echo "✅ Stage 2 complete"
echo ""

# Stage 3: Full complexity (all experiments)
echo "📊 Stage 3: Full Complexity"
echo "   Experiments: All (standard_enhanced, residual_listwise, aggressive_reg)"
echo "   Adds: Large batch + aggressive regularization"
uv run scripts/train_flexible_opportunistic.py \
    --data "$DATA_FILE" \
    --max_experiments 3 \
    --train_split 0.8 \
    2>&1 | tee "${OUTPUT_DIR}/stage3_full.log"

echo ""
echo "✅ All stages complete!"
echo ""
echo "Results saved to: $OUTPUT_DIR"

