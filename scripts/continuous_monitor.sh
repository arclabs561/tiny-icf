#!/bin/bash
# Continuous monitoring loop for all research-aligned experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

INTERVAL="${1:-60}"  # Default 60 seconds

echo "🔄 Continuous Monitoring (every ${INTERVAL}s)"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "📊 Research-Aligned Experiments Status"
    echo "======================================"
    echo "$(date)"
    echo ""
    
    # Quick status
    ./scripts/monitor_research_aligned_experiments.sh
    
    echo ""
    echo "📈 Latest Metrics:"
    for exp in research_aligned_standard research_aligned_neural_sort research_aligned_high_spearman research_aligned_strong_reg research_aligned_residual; do
        metrics_file="models/${exp}/lightning_logs/version_0/metrics.csv"
        if [ -f "$metrics_file" ]; then
            # Try to get latest validation metrics
            latest=$(tail -1 "$metrics_file" 2>/dev/null)
            if echo "$latest" | grep -q "val_spearman_corr"; then
                epoch=$(echo "$latest" | cut -d',' -f1)
                spearman=$(echo "$latest" | awk -F',' '{for(i=1;i<=NF;i++) if($i ~ /val_spearman_corr/) print $(i+1)}' | head -1)
                if [ ! -z "$spearman" ] && [ "$spearman" != "nan" ]; then
                    echo "   ${exp}: Epoch ${epoch}, Spearman ${spearman}"
                else
                    echo "   ${exp}: Epoch ${epoch} (training...)"
                fi
            else
                echo "   ${exp}: Training (no validation yet)"
            fi
        elif [ -f "models/${exp}/training.log" ]; then
            echo "   ${exp}: Starting..."
        else
            echo "   ${exp}: Not started"
        fi
    done
    
    echo ""
    echo "⏳ Next update in ${INTERVAL} seconds..."
    sleep $INTERVAL
done

