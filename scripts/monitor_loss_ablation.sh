#!/bin/bash
# Monitor loss ablation experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

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

echo "📊 Loss Ablation Experiments Status"
echo "===================================="
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    EXP_DIR="models/$exp"
    LOG_FILE="$EXP_DIR/training.log"
    METRICS_FILE="$EXP_DIR/lightning_logs/version_0/metrics.csv"
    
    if [ ! -f "$LOG_FILE" ]; then
        echo "⏳ $exp: Not started"
        continue
    fi
    
    # Check if process is running
    if pgrep -f "$exp" > /dev/null; then
        STATUS="🟢 Running"
    else
        STATUS="🔴 Stopped"
    fi
    
    # Get latest metrics if available
    if [ -f "$METRICS_FILE" ]; then
        LATEST=$(tail -1 "$METRICS_FILE" 2>/dev/null | cut -d',' -f1)
        if [ -n "$LATEST" ]; then
            # Try to extract Spearman if available
            SPEARMAN=$(tail -1 "$METRICS_FILE" 2>/dev/null | grep -o 'val_spearman_corr,[0-9.]*' | cut -d',' -f2 || echo "N/A")
            if [ "$SPEARMAN" != "N/A" ] && [ -n "$SPEARMAN" ]; then
                STATUS="$STATUS | Spearman: $SPEARMAN"
            fi
        fi
    fi
    
    # Get log size
    LOG_SIZE=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
    STATUS="$STATUS | Log lines: $LOG_SIZE"
    
    echo "$STATUS - $exp"
done

echo ""
echo "📊 Quick Analysis:"
echo "   uv run python scripts/analyze_loss_ablation_results.py"
echo ""
echo "📝 View logs:"
echo "   tail -f models/loss_ablation_*/training.log"

