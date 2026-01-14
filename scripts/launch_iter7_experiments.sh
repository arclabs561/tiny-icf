#!/bin/bash
# Launch Iteration 7 experiments in parallel

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FILE="$PROJECT_ROOT/data/word_frequency.csv"
TRAIN_SCRIPT="$PROJECT_ROOT/../trainctl/training/scripts/train_flexible_opportunistic.py"

# Check if Iter7 experiments JSON exists
ITER7_JSON="$PROJECT_ROOT/models/iter7_experiments.json"
if [ ! -f "$ITER7_JSON" ]; then
    echo "❌ Iter7 experiments JSON not found at $ITER7_JSON"
    echo "   Run: python3 scripts/plan_iter7_experiments.py"
    exit 1
fi

# Extract experiment names from JSON
EXPERIMENTS=$(python3 -c "
import json
import sys
from pathlib import Path

json_path = Path('$ITER7_JSON')
with open(json_path, 'r') as f:
    experiments = json.load(f)

for exp in experiments:
    print(exp['name'])
")

echo "🚀 Launching Iteration 7 Experiments"
echo "======================================"
echo ""

# Launch each experiment in background
for exp_name in $EXPERIMENTS; do
    echo "📦 Launching: $exp_name"
    # Create experiment directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/models/${exp_name}"
    # Use uv run to ensure dependencies are available
    nohup uv run "$TRAIN_SCRIPT" \
        --data "$DATA_FILE" \
        --experiments "$exp_name" \
        > "$PROJECT_ROOT/models/${exp_name}/training.log" 2>&1 &
    echo "   PID: $!"
    sleep 2  # Stagger launches slightly
done

echo ""
echo "✅ All Iter7 experiments launched"
echo "   Monitor with: python3 scripts/simple_status_check.py --follow"
echo "   Or check logs: tail -f models/iter7_*/training.log"

