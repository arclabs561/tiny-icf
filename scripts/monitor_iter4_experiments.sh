#!/bin/bash
# Monitor Iteration 4 experiments

echo "📊 Iteration 4 Experiment Status"
echo "================================="
echo ""

# Check active processes
ACTIVE=$(ps aux | grep -E "iter4|train_flexible" | grep -v grep | wc -l | xargs)
echo "🔄 Active processes: $ACTIVE"
echo ""

# Check each experiment
for exp_dir in models/iter4_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        log_file="$exp_dir/training.log"
        
        if [ -f "$log_file" ]; then
            # Get latest epoch and spearman
            latest=$(tail -50 "$log_file" 2>/dev/null | grep -E "Epoch.*spearman" | tail -1)
            if [ -n "$latest" ]; then
                echo "✅ $exp_name: $latest"
            else
                # Check if it's starting
                if grep -q "Starting\|Initializing\|Loading" "$log_file" 2>/dev/null; then
                    echo "⏳ $exp_name: Starting..."
                elif grep -q "error\|Error\|Traceback" "$log_file" 2>/dev/null; then
                    echo "❌ $exp_name: Error detected (check log)"
                else
                    echo "⏳ $exp_name: In progress..."
                fi
            fi
        else
            echo "⏸️  $exp_name: Not started"
        fi
    fi
done

echo ""
echo "📝 View logs: tail -f models/iter4_*/training.log"
echo "🛑 Stop all: pkill -f train_flexible_opportunistic"

