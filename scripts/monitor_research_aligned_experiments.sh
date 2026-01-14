#!/bin/bash
# Monitor research-aligned experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "📊 Monitoring Research-Aligned Experiments"
echo "=========================================="
echo ""

EXPERIMENTS=(
    "research_aligned_standard"
    "research_aligned_neural_sort"
    "research_aligned_high_spearman"
    "research_aligned_strong_reg"
    "research_aligned_residual"
)

for exp in "${EXPERIMENTS[@]}"; do
    exp_dir="models/${exp}"
    
    if [ ! -d "$exp_dir" ]; then
        echo "⏳ $exp: Not started"
        continue
    fi
    
    # Check for metrics
    metrics_csv="${exp_dir}/lightning_logs/version_0/metrics.csv"
    if [ -f "$metrics_csv" ]; then
        # Get latest Spearman
        latest_spearman=$(tail -n 1 "$metrics_csv" | cut -d',' -f$(head -n 1 "$metrics_csv" | tr ',' '\n' | grep -n "val_spearman_corr" | cut -d: -f1) 2>/dev/null || echo "N/A")
        echo "✅ $exp: Running (latest Spearman: $latest_spearman)"
    elif [ -f "${exp_dir}/training.log" ]; then
        echo "🔄 $exp: In progress (check ${exp_dir}/training.log)"
    else
        echo "⏳ $exp: Directory exists but no metrics yet"
    fi
done

echo ""
echo "📈 For detailed comparison:"
echo "   uv run python scripts/compare_baseline_vs_research_aligned.py"
echo ""
echo "📋 For experiment registry:"
echo "   uv run python scripts/create_experiment_registry.py"
echo ""

