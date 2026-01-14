#!/bin/bash
# Monitor Iteration 6 experiments

PROJECT_ROOT="/Users/arc/Documents/dev/idf-est"
LOG_DIR="$PROJECT_ROOT/models/iter6_logs"

echo "📊 Iteration 6 Experiments Status"
echo "=================================="
echo ""

# Check active processes
ACTIVE=$(pgrep -f "train_flexible_opportunistic.*iter6" | wc -l | tr -d ' ')
echo "🔄 Active processes: $ACTIVE"
echo ""

# Check each experiment
if [ -d "$LOG_DIR" ]; then
    for log_file in "$LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            exp_name=$(basename "$log_file" .log)
            echo "📈 $exp_name:"
            
            # Extract latest status
            if grep -q "Epoch" "$log_file" 2>/dev/null; then
                tail -5 "$log_file" | grep -E "Epoch|Spearman|loss" | tail -1
            else
                echo "   Status: Starting..."
            fi
            echo ""
        fi
    done
else
    echo "⚠️  Log directory not found: $LOG_DIR"
fi

