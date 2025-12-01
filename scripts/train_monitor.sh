#!/bin/bash
# Idempotent training script: checks status, trains if needed, monitors, returns results
# Usage: train_monitor.sh [pod-id]

set -e

POD_ID="${1:-w4857gh85qldts}"
WORK_DIR="/workspace/tiny-icf"
MODEL_PATH="$WORK_DIR/models/model_runpod.pt"
TRAINING_LOG="$WORK_DIR/training.log"
LOCK_FILE="$WORK_DIR/.training.lock"

echo "🚀 Idempotent Training Monitor"
echo "================================"
echo "Pod: $POD_ID"
echo ""

# Function to check if training is running
check_training_status() {
    # Check for lock file
    if runpodctl exec python "$POD_ID" -c "import os; exit(0 if os.path.exists('$LOCK_FILE') else 1)" 2>/dev/null; then
        return 0  # Lock exists
    fi
    return 1  # No lock
}

# Function to check if model exists and is complete
check_model_complete() {
    if runpodctl exec python "$POD_ID" -c "import os, torch; exit(0 if os.path.exists('$MODEL_PATH') and os.path.getsize('$MODEL_PATH') > 1000 else 1)" 2>/dev/null; then
        return 0  # Model exists and is valid
    fi
    return 1  # Model doesn't exist or is invalid
}

# Function to start training
start_training() {
    echo "📤 Uploading project files..."
    # Note: runpodctl send syntax may vary - adjust as needed
    # For now, we'll assume files are uploaded or use SSH
    
    echo "🔧 Installing dependencies..."
    runpodctl exec python "$POD_ID" -c "
import subprocess
import sys
import os
os.chdir('$WORK_DIR')
result = subprocess.run(['uv', 'pip', 'install', '-e', '.'], 
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)
" || {
        echo "⚠️  Dependencies may already be installed, continuing..."
    }
    
    echo "🎯 Starting training..."
    runpodctl exec python "$POD_ID" -c "
import subprocess
import os
import sys
from pathlib import Path

os.chdir('$WORK_DIR')
Path('$LOCK_FILE').touch()

# Start training in background
cmd = [
    'python', '-m', 'tiny_icf.train_curriculum',
    '--data', 'data/multilingual/multilingual_combined.csv',
    '--typo-corpus', 'data/typos/github_typos.csv',
    '--multilingual',
    '--epochs', '50',
    '--batch-size', '128',
    '--lr', '1e-3',
    '--curriculum-stages', '5',
    '--output', '$MODEL_PATH'
]

with open('$TRAINING_LOG', 'w') as log:
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f'Training started with PID: {process.pid}')
    process.wait()
    Path('$LOCK_FILE').unlink(missing_ok=True)
" &
    
    echo "✅ Training started in background"
}

# Function to monitor training
monitor_training() {
    echo ""
    echo "📊 Monitoring training progress..."
    echo "Press Ctrl+C to stop monitoring (training continues)"
    echo ""
    
    while true; do
        # Check if training is still running
        if ! check_training_status; then
            if check_model_complete; then
                echo ""
                echo "✅ Training complete!"
                break
            else
                echo ""
                echo "⚠️  Training stopped but model not found"
                break
            fi
        fi
        
        # Show recent log output
        echo "--- $(date +%H:%M:%S) ---"
        runpodctl exec python "$POD_ID" -c "
import os
log_path = '$TRAINING_LOG'
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
        # Show last 10 lines
        for line in lines[-10:]:
            print(line.rstrip())
" 2>/dev/null || echo "Waiting for log file..."
        
        sleep 10
    done
}

# Function to show results
show_results() {
    echo ""
    echo "📈 Training Results"
    echo "==================="
    
    # Check model file
    if check_model_complete; then
        echo "✅ Model saved: $MODEL_PATH"
        
        # Get model size
        runpodctl exec python "$POD_ID" -c "
import os
size = os.path.getsize('$MODEL_PATH')
print(f'Model size: {size / 1024:.2f} KB')
" 2>/dev/null
        
        # Show final training metrics
        echo ""
        echo "Final training metrics:"
        runpodctl exec python "$POD_ID" -c "
import os
log_path = '$TRAINING_LOG'
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
        # Find last epoch info
        for line in reversed(lines):
            if 'Epoch' in line or 'Val Loss' in line or 'Train Loss' in line:
                print(line.rstrip())
                if 'Val Loss' in line:
                    break
" 2>/dev/null || echo "Log file not found"
        
        echo ""
        echo "🎉 Training successful!"
        echo ""
        echo "Next steps:"
        echo "1. Download model: runpodctl receive $POD_ID $MODEL_PATH ./models/"
        echo "2. Validate: python scripts/validate_trained_model.py --model $MODEL_PATH"
    else
        echo "❌ Model not found or incomplete"
        exit 1
    fi
}

# Main execution
main() {
    # Check if model already exists and is complete
    if check_model_complete; then
        echo "✅ Model already exists and appears complete"
        echo ""
        read -p "Re-train? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            show_results
            exit 0
        fi
    fi
    
    # Check if training is already running
    if check_training_status; then
        echo "⏳ Training already in progress"
        monitor_training
        show_results
    else
        # Start training
        start_training
        sleep 5  # Wait a moment for training to start
        monitor_training
        show_results
    fi
}

main "$@"

