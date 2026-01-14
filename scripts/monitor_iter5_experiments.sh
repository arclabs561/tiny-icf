#!/bin/bash
# Monitor Iteration 5 experiments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

echo "📊 Iteration 5 Experiment Status"
echo "================================="
echo ""

# Check active processes
active_count=$(pgrep -f "train_flexible_opportunistic.*iter5" | wc -l | tr -d ' ')
echo "🔄 Active processes: $active_count"
echo ""

# Check each experiment
for exp_dir in "$MODELS_DIR"/iter5_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        log_file="$exp_dir/training.log"
        
        if [ -f "$log_file" ]; then
            # Extract latest epoch info
            latest=$(tail -20 "$log_file" | grep -E "Epoch [0-9]+:" | tail -1)
            if [ -n "$latest" ]; then
                echo "✅ $exp_name: $latest"
            else
                echo "⏳ $exp_name: Starting..."
            fi
        else
            echo "⏳ $exp_name: Not started yet"
        fi
    fi
done

echo ""
echo "📝 View logs: tail -f models/iter5_*/training.log"
echo "🛑 Stop all: pkill -f train_flexible_opportunistic"

