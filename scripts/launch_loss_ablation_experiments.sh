#!/bin/bash
# Launch systematic loss ablation experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Experiments to launch
EXPERIMENTS=(
    "loss_ablation_pure_spearman"
    "loss_ablation_pure_ranking"
    "loss_ablation_balanced_hybrid"
    "loss_ablation_high_spearman"
    "loss_ablation_very_high_spearman"
    "loss_ablation_high_ranking"
    "loss_ablation_no_focal"
    "loss_ablation_with_monotonicity"
    "loss_ablation_low_spearman"
    "loss_ablation_equal_weights"
)

echo "🚀 Launching Loss Ablation Experiments"
echo "======================================"
echo ""

# Check if data file exists
DATA_FILE="${1:-data/word_frequency.csv}"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Data file not found: $DATA_FILE"
    echo "Usage: $0 [data_file.csv]"
    exit 1
fi

# Launch each experiment
for exp in "${EXPERIMENTS[@]}"; do
    echo "📊 Launching: $exp"
    
    EXP_DIR="models/$exp"
    mkdir -p "$EXP_DIR"
    
    LOG_FILE="$EXP_DIR/training.log"
    
    # Launch in background
    (
        cd "$PROJECT_ROOT"
        uv run python ../trainctl/training/scripts/train_flexible_opportunistic.py \
            --data "$DATA_FILE" \
            --experiments "$exp" \
            > "$LOG_FILE" 2>&1
    ) &
    
    echo "   ✅ Started (PID: $!)"
    echo "   📝 Log: $LOG_FILE"
    echo ""
    
    # Small delay to avoid resource contention
    sleep 2
done

echo "✅ All experiments launched!"
echo ""
echo "📊 Monitor with:"
echo "   python scripts/compare_all_experiments.py"
echo "   python scripts/analyze_all_experiments.py"
echo ""
echo "🔍 Check logs:"
echo "   tail -f models/loss_ablation_*/training.log"

