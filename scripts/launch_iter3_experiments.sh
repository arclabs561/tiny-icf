#!/bin/bash
# Launch Iteration 3 experiments based on loss ablation results

DATA_FILE="${1:-data/word_frequency.csv}"
TRAINCTL_SCRIPT="../trainctl/training/scripts/train_flexible_opportunistic.py"

EXPERIMENTS=(
    'iter3_finetune_8x_06x'
    'iter3_finetune_12x_04x'
    'iter3_finetune_10x_07x'
    'iter3_neural_sort'
    'iter3_probabilistic'
    'iter3_distillation_best_loss'
    'iter3_longer_training'
    'iter3_residual_architecture'
    'iter3_refined_focal'
)

echo "🚀 Launching Iteration 3 Experiments"
echo "====================================="
echo ""
echo "📊 Based on loss ablation results:"
echo "   Best: balanced_hybrid (0.1649 Spearman)"
echo "   Config: 10× Spearman + 0.5× Ranking"
echo ""
echo "🎯 Iteration 3 Focus:"
echo "   1. Fine-tune weights around best config"
echo "   2. Test different ranking methods (NeuralSort, Probabilistic)"
echo "   3. Combine best loss with ModernBERT distillation"
echo "   4. Longer training (200 epochs)"
echo "   5. Different architecture (ResidualICF)"
echo "   6. Refined focal loss"
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

echo "✅ All Iteration 3 experiments launched!"
echo ""
echo "📊 Monitor with:"
echo "   tail -f models/iter3_*/training.log"
echo "   ./scripts/continuous_monitor_all.sh 60"
echo ""
echo "🛑 Stop all with:"
echo "   pkill -f train_flexible_opportunistic"

