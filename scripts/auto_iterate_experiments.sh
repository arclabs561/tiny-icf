#!/bin/bash
# Automatically iterate on experiments: launch, monitor, analyze, improve

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

EXPERIMENT="${1:-research_aligned_standard}"
DATA_FILE="${2:-data/word_frequency.csv}"
ITERATION="${3:-1}"

echo "🔄 Auto-Iterate: $EXPERIMENT (Iteration $ITERATION)"
echo "=================================================="
echo ""

# Step 1: Launch experiment
echo "1️⃣  Launching experiment..."
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Data file not found: $DATA_FILE"
    exit 1
fi

# Create iteration-specific name
ITERATION_NAME="${EXPERIMENT}_iter${ITERATION}"
echo "   Experiment name: $ITERATION_NAME"

# Launch (this would need to be modified to use iteration name)
# For now, just monitor existing
echo "   (Using existing experiment: $EXPERIMENT)"
echo ""

# Step 2: Monitor and wait for some progress
echo "2️⃣  Monitoring progress..."
MAX_WAIT=300  # 5 minutes
WAIT_INTERVAL=30
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if [ -f "models/${EXPERIMENT}/lightning_logs/version_0/metrics.csv" ]; then
        # Check if we have at least 5 epochs
        EPOCH_COUNT=$(tail -n +2 "models/${EXPERIMENT}/lightning_logs/version_0/metrics.csv" | wc -l | tr -d ' ')
        if [ "$EPOCH_COUNT" -ge 5 ]; then
            echo "   ✅ Progress detected: $EPOCH_COUNT epochs"
            break
        fi
    fi
    
    echo "   ⏳ Waiting for progress... (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "   ⚠️  Timeout waiting for progress"
    exit 1
fi

# Step 3: Analyze
echo ""
echo "3️⃣  Analyzing results..."
uv run python scripts/iterative_improve.py "$EXPERIMENT"

# Step 4: Show current metrics
echo ""
echo "4️⃣  Current Metrics:"
if [ -f "models/${EXPERIMENT}/lightning_logs/version_0/metrics.csv" ]; then
    tail -1 "models/${EXPERIMENT}/lightning_logs/version_0/metrics.csv" | \
        awk -F',' '{printf "   Epoch: %s, Val Spearman: %s, Val MAE: %s\n", $1, $4, $5}'
fi

echo ""
echo "✅ Iteration $ITERATION complete!"
echo ""
echo "📊 Next steps:"
echo "   - Monitor: ./scripts/monitor_research_aligned_experiments.sh"
echo "   - Analyze: uv run python scripts/iterative_improve.py $EXPERIMENT"
echo "   - Compare: uv run python scripts/compare_baseline_vs_research_aligned.py"

