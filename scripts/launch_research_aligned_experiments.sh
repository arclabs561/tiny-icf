#!/bin/bash
# Launch research-aligned loss experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINCTL_SCRIPT="$PROJECT_ROOT/../trainctl/training/scripts/train_flexible_opportunistic.py"

cd "$PROJECT_ROOT"

echo "🚀 Launching Research-Aligned Loss Experiments"
echo "=============================================="
echo ""

# List of research-aligned experiments
EXPERIMENTS=(
    "research_aligned_standard"
    "research_aligned_neural_sort"
    "research_aligned_high_spearman"
    "research_aligned_strong_reg"
    "research_aligned_residual"
)

# Check if data file exists
DATA_FILE="${1:-data/word_frequency.csv}"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Data file not found: $DATA_FILE"
    echo "   Usage: $0 [data_file.csv]"
    exit 1
fi

# Check if training script exists
if [ ! -f "$TRAINCTL_SCRIPT" ]; then
    echo "❌ Training script not found: $TRAINCTL_SCRIPT"
    exit 1
fi

# Test integration first
echo "🧪 Testing integration..."
if ! uv run python scripts/quick_test_research_aligned.py > /dev/null 2>&1; then
    echo "❌ Integration test failed. Run manually:"
    echo "   uv run python scripts/quick_test_research_aligned.py"
    exit 1
fi
echo "✅ Integration test passed"
echo ""

echo "📊 Data file: $DATA_FILE"
echo "📁 Experiments: ${#EXPERIMENTS[@]}"
echo ""

# Ask for confirmation if launching all
if [ "${2:-}" != "--yes" ]; then
    echo "⚠️  This will launch ${#EXPERIMENTS[@]} experiments in parallel."
    echo "   Press Ctrl+C to cancel, or wait 5 seconds to continue..."
    sleep 5
fi

# Create models directory
mkdir -p "$PROJECT_ROOT/models"

# Launch each experiment
PIDS=()
for exp in "${EXPERIMENTS[@]}"; do
    EXP_DIR="$PROJECT_ROOT/models/${exp}"
    mkdir -p "$EXP_DIR"
    
    echo "▶️  Launching: $exp"
    echo "   Command: uv run python $TRAINCTL_SCRIPT --data $DATA_FILE --experiments $exp"
    
    # Launch in background
    uv run python "$TRAINCTL_SCRIPT" \
        --data "$DATA_FILE" \
        --experiments "$exp" \
        > "$EXP_DIR/training.log" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    echo "   PID: $PID"
    echo "   Log: $EXP_DIR/training.log"
    echo ""
    
    # Small delay between launches to avoid resource contention
    sleep 3
done

echo "✅ All experiments launched!"
echo ""
echo "📊 PIDs: ${PIDS[@]}"
echo ""
echo "📊 Monitor progress:"
echo "   - Quick check: ./scripts/monitor_research_aligned_experiments.sh"
echo "   - Check logs: tail -f models/research_aligned_*/training.log"
echo "   - Check metrics: cat models/research_aligned_*/lightning_logs/version_0/metrics.csv"
echo "   - Compare baselines: uv run python scripts/compare_baseline_vs_research_aligned.py"
echo "   - Update registry: uv run python scripts/create_experiment_registry.py"
echo ""
echo "🛑 To stop all experiments:"
echo "   kill ${PIDS[@]}"
echo ""

