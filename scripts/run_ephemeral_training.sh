#!/bin/bash
# Robust training script for ephemeral RunPod environments
# Handles pod restarts, checkpointing, and auto-resume

set -e

SSH_HOST="213.173.111.79"
SSH_PORT="34185"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/root/idf-est"
LOG_FILE="training_ephemeral.log"

echo "=== Ephemeral Training Setup ==="
echo "Host: $SSH_HOST:$SSH_PORT"
echo "Remote Dir: $REMOTE_DIR"
echo ""

# Function to check if training is already running
check_running() {
    ssh -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST" \
        "pgrep -f 'train_ephemeral_robust.py' > /dev/null 2>&1"
}

# Function to start training
start_training() {
    echo "🚀 Starting training on remote server..."
    ssh -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST" << 'ENDSSH'
cd /root/idf-est

# Activate environment if needed
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if data exists
if [ ! -f "data/word_frequency.csv" ]; then
    echo "⚠️  Data file not found. Please ensure data/word_frequency.csv exists."
    exit 1
fi

# Start training in background with nohup
nohup python3 scripts/train_ephemeral_robust.py \
    --data data/word_frequency.csv \
    --output-dir models \
    --epochs 200 \
    --batch-size 256 \
    --lr 1e-3 \
    --rank-weight 5.0 \
    --early-stop-patience 20 \
    --checkpoint-interval 1 \
    > training_ephemeral.log 2>&1 &

echo $! > training_ephemeral.pid
echo "✓ Training started (PID: $(cat training_ephemeral.pid))"
echo "📝 Logs: tail -f training_ephemeral.log"
ENDSSH
}

# Function to monitor training
monitor_training() {
    echo "📊 Monitoring training..."
    ssh -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST" << 'ENDSSH'
cd /root/idf-est

if [ -f training_ephemeral.pid ]; then
    PID=$(cat training_ephemeral.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Training is running (PID: $PID)"
        echo ""
        echo "=== Recent Logs ==="
        tail -20 training_ephemeral.log
        echo ""
        echo "=== Checkpoint Status ==="
        if [ -f models/checkpoint_ephemeral_robust.pt ]; then
            ls -lh models/checkpoint_ephemeral_robust.pt
            echo "✓ Checkpoint exists"
        else
            echo "⚠️  No checkpoint yet"
        fi
    else
        echo "⚠️  Training process not found (may have completed or crashed)"
    fi
else
    echo "⚠️  PID file not found"
fi
ENDSSH
}

# Function to resume training
resume_training() {
    echo "🔄 Resuming training from checkpoint..."
    ssh -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST" << 'ENDSSH'
cd /root/idf-est

if [ -d "venv" ]; then
    source venv/bin/activate
fi

nohup python3 scripts/train_ephemeral_robust.py \
    --data data/word_frequency.csv \
    --output-dir models \
    --epochs 200 \
    --batch-size 256 \
    --lr 1e-3 \
    --rank-weight 5.0 \
    --early-stop-patience 20 \
    --checkpoint-interval 1 \
    --resume models/checkpoint_ephemeral_robust.pt \
    > training_ephemeral.log 2>&1 &

echo $! > training_ephemeral.pid
echo "✓ Training resumed (PID: $(cat training_ephemeral.pid))"
ENDSSH
}

# Main logic
case "${1:-monitor}" in
    start)
        if check_running; then
            echo "⚠️  Training already running"
            monitor_training
        else
            start_training
        fi
        ;;
    resume)
        resume_training
        ;;
    monitor|status)
        monitor_training
        ;;
    stop)
        echo "🛑 Stopping training..."
        ssh -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST" \
            "pkill -f 'train_ephemeral_robust.py' && rm -f training_ephemeral.pid && echo '✓ Training stopped'"
        ;;
    logs)
        ssh -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST" \
            "cd $REMOTE_DIR && tail -f training_ephemeral.log"
        ;;
    *)
        echo "Usage: $0 {start|resume|monitor|stop|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start new training session"
        echo "  resume  - Resume from checkpoint"
        echo "  monitor - Check training status"
        echo "  stop    - Stop training"
        echo "  logs    - Follow training logs"
        exit 1
        ;;
esac

