#!/bin/bash
# Quick script to start flexible training on remote pod

SSH_HOST="${1:-root@194.68.245.50}"
SSH_PORT="${2:-22106}"
SSH_KEY="${3:-~/.ssh/id_ed25519}"

echo "Starting flexible training on $SSH_HOST:$SSH_PORT..."

ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" << 'EOF'
cd /root/idf-est
source .venv/bin/activate

# Kill any existing training
if [ -f training.pid ]; then
    OLD_PID=$(cat training.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Stopping existing training (PID: $OLD_PID)..."
        kill $OLD_PID 2>/dev/null || true
        sleep 2
    fi
fi

# Start new training (with unbuffered Python output)
echo "Starting training..."
nohup uv run scripts/train_flexible_opportunistic.py \
    --max_experiments 3 \
    --train_split 0.8 \
    > training.log 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > training.pid
echo "✓ Training started with PID: $TRAIN_PID"

# Show initial output
sleep 3
echo ""
echo "=== Training Log (last 20 lines) ==="
tail -20 training.log
EOF

echo ""
echo "✓ Training started!"
echo "Monitor with: bash scripts/monitor_flexible_training.sh"

