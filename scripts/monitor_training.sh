#!/bin/bash
# Monitor training progress without hanging

LOG_FILE="${1:-training.log}"
INTERVAL="${2:-10}"

while true; do
    if [ -f "$LOG_FILE" ]; then
        clear
        echo "=== Training Progress (last updated: $(date +%H:%M:%S)) ==="
        echo ""
        tail -30 "$LOG_FILE" | grep -E "(Epoch|Stage|loss|Saved|complete|Error|Using device|Model parameters)" | tail -20
        echo ""
        echo "=== Process Status ==="
        ps aux | grep "[p]ython.*train_curriculum" | head -1 || echo "Training process not found"
        echo ""
        echo "=== Model Files ==="
        ls -lh models/*.pt 2>/dev/null | tail -3 || echo "No models yet"
    else
        echo "Waiting for $LOG_FILE to be created..."
    fi
    sleep "$INTERVAL"
done

