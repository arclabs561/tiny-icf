#!/bin/bash
# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
"""
Monitor flexible training on remote pod.
"""

SSH_HOST="${1:-root@194.68.245.50}"
SSH_PORT="${2:-22106}"
SSH_KEY="${3:-~/.ssh/id_ed25519}"

echo "=========================================="
echo "Monitoring Training"
echo "=========================================="
echo "Host: $SSH_HOST"
echo ""

ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" << 'EOF'
cd /root/idf-est

echo "=== Training Processes ==="
ps aux | grep -E "train_flexible|python.*train" | grep -v grep || echo "No training processes found"

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"

echo ""
echo "=== Recent Training Log ==="
if [ -f "training.log" ]; then
    tail -30 training.log
else
    echo "No training.log found"
fi

echo ""
echo "=== Model Checkpoints ==="
if [ -d "models" ]; then
    find models -name "*.pt" -o -name "*.json" | head -10
    echo ""
    echo "=== Experiment Summary ==="
    if [ -f "models/experiment_summary.json" ]; then
        cat models/experiment_summary.json | python3 -m json.tool 2>/dev/null || cat models/experiment_summary.json
    fi
else
    echo "No models directory"
fi

echo ""
echo "=== Disk Usage ==="
df -h / | tail -1
EOF

