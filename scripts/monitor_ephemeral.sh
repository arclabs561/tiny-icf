#!/bin/bash
# Enhanced monitoring script for ephemeral training
# Shows comprehensive status, progress, and resource usage

SSH_HOST="213.173.111.79"
SSH_PORT="34185"
SSH_KEY="$HOME/.ssh/id_ed25519"

ssh -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST" << 'ENDSSH'
cd /root/idf-est
export PATH="$HOME/.cargo/bin:$PATH"

echo "═══════════════════════════════════════════════════════════════"
echo "  EPHEMERAL TRAINING MONITOR"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Process Status
echo "📊 PROCESS STATUS"
echo "───────────────────────────────────────────────────────────────"
if [ -f training_ephemeral.pid ]; then
    PID=$(cat training_ephemeral.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Training running (PID: $PID)"
        
        # Resource usage
        ps -p $PID -o pid,pcpu,pmem,etime,cmd --no-headers | awk '{
            printf "  CPU: %s%%  |  Memory: %s%%  |  Runtime: %s\n", $2, $3, $4
        }'
        
        # Check for GPU usage if nvidia-smi available
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "🎮 GPU STATUS"
            echo "───────────────────────────────────────────────────────────────"
            nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F', ' '{
                printf "  %s\n", $1
                printf "  Usage: %s%%  |  Memory: %s/%s MB  |  Temp: %s°C\n", $2, $3, $4, $5
            }'
        fi
    else
        echo "⚠️  Process not found (PID: $PID)"
        echo "  Process may have completed or crashed"
    fi
else
    echo "⚠️  No PID file found"
fi

# Also check for any python training processes
TRAIN_PROCS=$(pgrep -f "train_ephemeral_robust.py" | wc -l)
if [ "$TRAIN_PROCS" -gt 0 ]; then
    echo ""
    echo "  Active training processes: $TRAIN_PROCS"
    pgrep -f "train_ephemeral_robust.py" | while read p; do
        ps -p $p -o pid,pcpu,pmem,etime --no-headers 2>/dev/null | awk '{
            printf "    PID %s: CPU %s%%, MEM %s%%, Runtime %s\n", $1, $2, $3, $4
        }'
    done
fi

echo ""

# Training Progress
echo "📈 TRAINING PROGRESS"
echo "───────────────────────────────────────────────────────────────"
if [ -f training_ephemeral.log ]; then
    # Extract current epoch
    CURRENT_EPOCH=$(grep -E "Epoch [0-9]+/[0-9]+" training_ephemeral.log | tail -1 | grep -oE "Epoch [0-9]+" | grep -oE "[0-9]+" || echo "?")
    TOTAL_EPOCHS=$(grep -E "Epoch [0-9]+/[0-9]+" training_ephemeral.log | tail -1 | grep -oE "/[0-9]+" | grep -oE "[0-9]+" || echo "?")
    
    if [ "$CURRENT_EPOCH" != "?" ]; then
        PROGRESS=$(echo "scale=1; $CURRENT_EPOCH * 100 / $TOTAL_EPOCHS" | bc 2>/dev/null || echo "?")
        echo "  Current: Epoch $CURRENT_EPOCH / $TOTAL_EPOCHS ($PROGRESS%)"
    fi
    
    # Extract best Spearman
    BEST_SPEARMAN=$(grep -E "Best.*Spearman|best_spearman|Saved best model" training_ephemeral.log | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "?")
    if [ "$BEST_SPEARMAN" != "?" ]; then
        echo "  Best Spearman: $BEST_SPEARMAN"
    fi
    
    # Extract latest metrics
    LATEST_METRICS=$(grep -E "Train.*Loss.*Spearman|Val.*Loss.*Spearman" training_ephemeral.log | tail -1)
    if [ -n "$LATEST_METRICS" ]; then
        echo "  Latest: $LATEST_METRICS"
    fi
    
    # Training time
    START_TIME=$(stat -c %Y training_ephemeral.log 2>/dev/null || stat -f %m training_ephemeral.log 2>/dev/null)
    if [ -n "$START_TIME" ]; then
        ELAPSED=$(($(date +%s) - $START_TIME))
        HOURS=$((ELAPSED / 3600))
        MINS=$(((ELAPSED % 3600) / 60))
        echo "  Training time: ${HOURS}h ${MINS}m"
    fi
else
    echo "  No log file found"
fi

echo ""

# Checkpoint Status
echo "💾 CHECKPOINT STATUS"
echo "───────────────────────────────────────────────────────────────"
if [ -f models/checkpoint_ephemeral_robust.pt ]; then
    CHECKPOINT_SIZE=$(du -h models/checkpoint_ephemeral_robust.pt | cut -f1)
    CHECKPOINT_TIME=$(stat -c %y models/checkpoint_ephemeral_robust.pt 2>/dev/null | cut -d' ' -f1-2 || stat -f "%Sm" models/checkpoint_ephemeral_robust.pt 2>/dev/null)
    echo "✓ Checkpoint exists"
    echo "  Size: $CHECKPOINT_SIZE"
    echo "  Last saved: $CHECKPOINT_TIME"
    
    # Try to extract epoch from checkpoint (if possible)
    python3 -c "
import torch
try:
    ckpt = torch.load('models/checkpoint_ephemeral_robust.pt', map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', '?')
    best = ckpt.get('best_spearman', '?')
    print(f'  Epoch: {epoch}')
    print(f'  Best Spearman: {best:.4f}' if best != '?' else '  Best Spearman: ?')
except Exception as e:
    print(f'  (Could not read checkpoint: {e})')
" 2>/dev/null || echo "  (Could not read checkpoint details)"
else
    echo "⚠️  No checkpoint yet"
fi

echo ""

# Model Files
echo "📦 MODEL FILES"
echo "───────────────────────────────────────────────────────────────"
if ls models/*.pt 2>/dev/null | grep -q .; then
    ls -lh models/*.pt 2>/dev/null | tail -5 | awk '{
        printf "  %s  %s  %s\n", $5, $6" "$7" "$8, $9
    }'
else
    echo "  No model files yet"
fi

echo ""

# Recent Logs (key information only)
echo "📝 RECENT ACTIVITY"
echo "───────────────────────────────────────────────────────────────"
if [ -f training_ephemeral.log ]; then
    # Show last few important lines
    tail -15 training_ephemeral.log | grep -E "(Epoch|Spearman|Loss|Saved|Error|Traceback|✓|⚠️)" | tail -8 || tail -8 training_ephemeral.log
else
    echo "  No log file"
fi

echo ""

# System Resources
echo "🖥️  SYSTEM RESOURCES"
echo "───────────────────────────────────────────────────────────────"
echo "  CPU Load: $(uptime | awk -F'load average:' '{print $2}')"
echo "  Memory: $(free -h | grep Mem | awk '{printf "%.1f%% used (%s / %s)", $3/$2*100, $3, $2}')"
echo "  Disk: $(df -h /root/idf-est | tail -1 | awk '{print $5 " used (" $3 " / " $2 ")"}')"

echo ""

# Recommendations
echo "💡 RECOMMENDATIONS"
echo "───────────────────────────────────────────────────────────────"
if [ ! -f training_ephemeral.pid ] || ! ps -p $(cat training_ephemeral.pid 2>/dev/null) > /dev/null 2>&1; then
    if [ -f models/checkpoint_ephemeral_robust.pt ]; then
        echo "  → Resume training: python3 scripts/train_ephemeral_robust.py ... --resume models/checkpoint_ephemeral_robust.pt"
    else
        echo "  → Start training: python3 scripts/train_ephemeral_robust.py ..."
    fi
else
    echo "  → Training is running - monitor with: tail -f training_ephemeral.log"
    echo "  → Check progress: ./scripts/monitor_ephemeral.sh"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
ENDSSH
