#!/bin/bash
# Monitoring script for RunPod web terminal

cd /root/idf-est

echo "=== Ablation Study Monitor ==="
echo ""

# Check if running
if [ -f logs/ablation.pid ]; then
    PID=$(cat logs/ablation.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Process running (PID: $PID)"
    else
        echo "✗ Process not running (PID: $PID)"
    fi
else
    echo "? No PID file found"
fi

echo ""

# Check status file
if [ -f logs/ablation_status.txt ]; then
    echo "=== Status ==="
    cat logs/ablation_status.txt
    echo ""
fi

# Check results
if [ -f ablation_results.json ]; then
    echo "=== Results File ==="
    python3 -c "
import json
try:
    with open('ablation_results.json') as f:
        d = json.load(f)
    print(f'Configs completed: {len(d)}')
    for k, v in d.items():
        spearman = v.get('final_spearman', 'N/A')
        epochs = v.get('epochs_completed', 0)
        if isinstance(spearman, (int, float)):
            print(f'  {k}: spearman={spearman:.4f}, epochs={epochs}')
        else:
            print(f'  {k}: spearman={spearman}, epochs={epochs}')
except Exception as e:
    print(f'Error reading results: {e}')
" 2>/dev/null || echo "Results file exists but not readable yet"
    echo ""
fi

# Check GPU
echo "=== GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo ""

# Check log size
if [ -f logs/ablation_output.log ]; then
    echo "=== Log Info ==="
    echo "Size: $(du -h logs/ablation_output.log | cut -f1)"
    echo "Lines: $(wc -l < logs/ablation_output.log)"
    echo "Last modified: $(stat -c '%y' logs/ablation_output.log 2>/dev/null || stat -f '%Sm' logs/ablation_output.log 2>/dev/null)"
    echo ""
    echo "=== Last 20 Lines ==="
    tail -20 logs/ablation_output.log
else
    echo "No output log yet"
fi

# Check Aim
echo ""
echo "=== Aim Status ==="
if command -v aim &> /dev/null; then
    echo "✓ Aim installed"
    if [ -d .aim ]; then
        echo "✓ Aim repo exists"
        aim status 2>/dev/null || echo "Aim repo initialized"
    else
        echo "? Aim repo not initialized"
    fi
else
    echo "✗ Aim not installed"
    echo "  Install: uv pip install aim"
fi

