#!/bin/bash
# Continuous iteration: launch, monitor, analyze, improve, repeat

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

DATA_FILE="${1:-data/word_frequency.csv}"
ITERATION="${2:-1}"

echo "🔄 Continuous Iteration Workflow"
echo "================================="
echo ""

# Phase 1: Launch initial batch
if [ "$ITERATION" == "1" ]; then
    echo "📦 Phase 1: Launching initial batch (5 experiments)"
    ./scripts/launch_research_aligned_experiments.sh "$DATA_FILE" --yes
    
    echo ""
    echo "⏳ Waiting for initial progress (30 minutes)..."
    sleep 1800  # 30 minutes
    
    echo ""
    echo "📊 Checking progress..."
    ./scripts/monitor_research_aligned_experiments.sh
fi

# Phase 2: Analyze and create improvements
echo ""
echo "🔍 Phase 2: Analyzing results..."

# Analyze each experiment
for exp in research_aligned_standard research_aligned_neural_sort research_aligned_high_spearman research_aligned_strong_reg research_aligned_residual; do
    if [ -f "models/${exp}/lightning_logs/version_0/metrics.csv" ]; then
        echo ""
        echo "Analyzing: $exp"
        uv run python scripts/iterative_improve.py "$exp" || true
    fi
done

# Phase 3: Launch iteration 2 if results are promising
echo ""
echo "🚀 Phase 3: Launching iteration 2 experiments..."

ITERATION2_EXPERIMENTS=(
    "research_aligned_monotonicity"
    "research_aligned_high_lr"
    "research_aligned_probabilistic"
)

for exp in "${ITERATION2_EXPERIMENTS[@]}"; do
    echo "▶️  Launching: $exp"
    nohup uv run python ../trainctl/training/scripts/train_flexible_opportunistic.py \
        --data "$DATA_FILE" \
        --experiments "$exp" \
        > "models/${exp}/training.log" 2>&1 &
    echo "   PID: $!"
    sleep 3
done

echo ""
echo "✅ Iteration $ITERATION complete!"
echo ""
echo "📊 Next iteration:"
echo "   $0 $DATA_FILE $((ITERATION + 1))"
echo ""
echo "📈 Monitor:"
echo "   ./scripts/monitor_research_aligned_experiments.sh"

