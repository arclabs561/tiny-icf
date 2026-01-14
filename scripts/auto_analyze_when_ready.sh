#!/bin/bash
# Automatically analyze loss ablation results when experiments reach sufficient epochs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MIN_EPOCHS=${1:-20}  # Minimum epochs before analysis
CHECK_INTERVAL=${2:-300}  # Check every 5 minutes

echo "🔍 Auto-Analyzer for Loss Ablation Experiments"
echo "=============================================="
echo "Minimum epochs: $MIN_EPOCHS"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

while true; do
    echo "$(date): Checking experiments..."
    
    # Count experiments with sufficient epochs
    READY_COUNT=0
    TOTAL_COUNT=0
    
    for exp_dir in models/loss_ablation_*/lightning_logs/version_0/metrics.csv; do
        if [ -f "$exp_dir" ]; then
            TOTAL_COUNT=$((TOTAL_COUNT + 1))
            EPOCH_COUNT=$(tail -1 "$exp_dir" 2>/dev/null | cut -d',' -f1 | grep -E '^[0-9]+$' || echo "0")
            if [ "$EPOCH_COUNT" -ge "$MIN_EPOCHS" ]; then
                READY_COUNT=$((READY_COUNT + 1))
            fi
        fi
    done
    
    echo "   Ready: $READY_COUNT / $TOTAL_COUNT experiments"
    
    # If at least 5 experiments are ready, run analysis
    if [ "$READY_COUNT" -ge 5 ]; then
        echo ""
        echo "✅ Sufficient experiments ready! Running analysis..."
        uv run python scripts/analyze_loss_ablation_results.py
        echo ""
        echo "💤 Waiting ${CHECK_INTERVAL}s before next check..."
    else
        echo "   ⏳ Waiting for more experiments to reach $MIN_EPOCHS epochs..."
    fi
    
    sleep "$CHECK_INTERVAL"
done

