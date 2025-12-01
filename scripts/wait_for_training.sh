#!/bin/bash
# Wait for training to complete and then run validation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

LOG_FILE="training.log"
CHECK_INTERVAL=30  # seconds

echo "Waiting for training to complete..."
echo "Checking every $CHECK_INTERVAL seconds"
echo ""

while true; do
    # Check if training process is running
    if ! ps aux | grep -q "[p]ython.*train_curriculum"; then
        echo "Training process not found - checking if complete..."
        
        # Check log for completion
        if grep -q "Training complete" "$LOG_FILE" 2>/dev/null; then
            echo "✓ Training complete!"
            break
        else
            echo "⚠ Training process stopped but log doesn't show completion"
            echo "  Last log entries:"
            tail -5 "$LOG_FILE" 2>/dev/null || echo "  (log file not found)"
            break
        fi
    fi
    
    # Show progress
    if [ -f "$LOG_FILE" ]; then
        last_epoch=$(grep -o "Epoch [0-9]*/30" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" | head -1)
        if [ -n "$last_epoch" ]; then
            progress=$((last_epoch * 100 / 30))
            echo "[$(date +%H:%M:%S)] Epoch $last_epoch/30 ($progress% complete)"
        fi
    fi
    
    sleep "$CHECK_INTERVAL"
done

echo ""
echo "Running post-training validation..."
./scripts/post_training_validation.sh

