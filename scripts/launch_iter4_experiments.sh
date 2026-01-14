#!/bin/bash
# Launch Iteration 4 experiments: Combining best components
# - Refined ResidualICF (pre-activation) architecture
# - ResearchAlignedICFLoss
# - Best config from Iter3 (12× Spearman + 0.4× Ranking)
# - Probabilistic ranking method

DATA_FILE="${1:-data/word_frequency.csv}"
TRAINCTL_SCRIPT="../trainctl/training/scripts/train_flexible_opportunistic.py"

EXPERIMENTS=(
    'iter4_residual_research_best'
    'iter4_residual_14x_03x'
    'iter4_residual_11x_05x'
    'iter4_residual_neural_sort'
    'iter4_residual_sigmoid'
    'iter4_residual_focal_high'
    'iter4_residual_monotonicity'
    'iter4_residual_distillation'
    'iter4_residual_adaptive_reg'
    'iter4_residual_longer'
    'iter4_residual_match_balanced'
)

echo "🚀 Launching Iteration 4 Experiments"
echo "====================================="
echo ""
echo "📊 Based on Iter3 results:"
echo "   Best Iter3: iter3_finetune_12x_04x (0.1811 Spearman)"
echo "   Previous best: residual_balanced (0.1864 Spearman)"
echo "   Best ranking method: Probabilistic (0.1809)"
echo ""
echo "🎯 Iteration 4 Strategy:"
echo "   1. Refined ResidualICF (pre-activation) + ResearchAlignedICFLoss"
echo "   2. Best config: 12× Spearman + 0.4× Ranking"
echo "   3. Probabilistic ranking method"
echo "   4. Test variations around best config"
echo "   5. Combine with distillation, monotonicity, adaptive reg"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    EXP_DIR="models/$exp"
    mkdir -p "$EXP_DIR"
    
    echo "📦 Launching: $exp"
    nohup uv run python "$TRAINCTL_SCRIPT" \
        --data "$DATA_FILE" \
        --experiments "$exp" \
        --max_experiments 1 \
        > "$EXP_DIR/training.log" 2>&1 &
    
    PID=$!
    echo "$PID" > "$EXP_DIR/training.pid"
    echo "   ✅ Launched (PID: $PID)"
    echo "   📝 Log: $EXP_DIR/training.log"
    echo ""
    
    sleep 2  # Stagger launches
done

echo "✅ All Iteration 4 experiments launched!"
echo ""
echo "📊 Monitor with:"
echo "   tail -f models/iter4_*/training.log"
echo "   ./scripts/continuous_monitor_all.sh 60"
echo ""
echo "🛑 Stop all with:"
echo "   pkill -f train_flexible_opportunistic"

