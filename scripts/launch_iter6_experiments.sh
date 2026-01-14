#!/bin/bash
# Launch Iteration 6 experiments: Transformer architectures + Fine-tuning

set -e

PROJECT_ROOT="/Users/arc/Documents/dev/idf-est"
DATA_FILE="$PROJECT_ROOT/data/word_frequency.csv"
TRAINCTL_SCRIPT="$PROJECT_ROOT/../trainctl/training/scripts/train_flexible_opportunistic.py"
EXPERIMENTS_JSON="$PROJECT_ROOT/models/iter6_experiments.json"

# Check if experiments file exists
if [ ! -f "$EXPERIMENTS_JSON" ]; then
    echo "❌ Experiments file not found: $EXPERIMENTS_JSON"
    echo "   Run: uv run python scripts/create_iter6_experiments.py"
    exit 1
fi

# Extract experiment names from JSON
EXPERIMENTS=$(python3 -c "
import json
with open('$EXPERIMENTS_JSON') as f:
    exps = json.load(f)
    print(' '.join([exp['name'] for exp in exps]))
")

echo "🚀 Launching Iteration 6 Experiments"
echo "======================================"
echo "Experiments: $EXPERIMENTS"
echo ""

# Create log directory
LOG_DIR="$PROJECT_ROOT/models/iter6_logs"
mkdir -p "$LOG_DIR"

# Launch each experiment in background
for exp in $EXPERIMENTS; do
    echo "📊 Launching: $exp"
    log_file="$LOG_DIR/${exp}.log"
    
    # Launch in background
    nohup uv run python "$TRAINCTL_SCRIPT" \
        --data "$DATA_FILE" \
        --experiments "$exp" \
        > "$log_file" 2>&1 &
    
    echo "   PID: $!"
    echo "   Log: $log_file"
    sleep 2  # Stagger launches
done

echo ""
echo "✅ All experiments launched!"
echo ""
echo "📊 Monitor with:"
echo "   tail -f $LOG_DIR/*.log"
echo ""
echo "🛑 Stop all with:"
echo "   pkill -f train_flexible_opportunistic"

