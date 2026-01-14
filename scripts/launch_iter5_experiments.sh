#!/bin/bash
# Launch Iteration 5 experiments in parallel
# Based on Iter4 findings: distillation (0.187) and adaptive_reg (0.178) are best

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINCTL_SCRIPT="$PROJECT_ROOT/../trainctl/training/scripts/train_flexible_opportunistic.py"
# Default data file (CSV format, as expected by training script)
DATA_FILE="$PROJECT_ROOT/data/word_frequency.csv"

# Iter5 experiments
EXPERIMENTS=(
    "iter5_distillation_adaptive_reg"
    "iter5_distillation_temp_20"
    "iter5_distillation_temp_30"
    "iter5_distillation_temp_40"
    "iter5_distillation_alpha_3"
    "iter5_distillation_alpha_5"
    "iter5_distillation_alpha_7"
    "iter5_distillation_longer"
    "iter5_adaptive_reg_longer"
    "iter5_distillation_adaptive_longer"
    "iter5_ratio_14x_3x"
    "iter5_ratio_11x_5x"
    "iter5_ratio_13x_4x"
    "iter5_focal_high"
    "iter5_monotonicity"
    "iter5_baseline_fixed_lr"
)

LOG_DIR="$PROJECT_ROOT/models/iter5_logs"
mkdir -p "$LOG_DIR"

echo "🚀 Launching Iteration 5 Experiments"
echo "======================================"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Log directory: $LOG_DIR"
echo ""

# Launch each experiment in background
for exp in "${EXPERIMENTS[@]}"; do
    log_file="$LOG_DIR/${exp}.log"
    echo "Launching $exp..."
    
    nohup uv run python "$TRAINCTL_SCRIPT" \
        --data "$DATA_FILE" \
        --experiments "$exp" \
        > "$log_file" 2>&1 &
    
    echo "  ✅ Started (PID: $!, log: $log_file)"
    sleep 2  # Small delay to avoid resource contention
done

echo ""
echo "✅ All experiments launched!"
echo ""
echo "📊 Monitor with:"
echo "   ./scripts/monitor_iter5_experiments.sh"
echo ""
echo "🛑 Stop all with:"
echo "   pkill -f train_flexible_opportunistic"

