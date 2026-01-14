#!/bin/bash
# RunPod ablation study - oneshot background execution

set -e

cd /root/idf-est
export PATH="$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1

# Kill any existing ablation processes
pkill -f 'ablation_loss_study' 2>/dev/null || true
sleep 2

# Create log directory
mkdir -p logs

# Parse arguments (for Aim tracking if provided)
AIM_ARGS=""
if [ "$1" == "--aim" ] || [ "$1" == "-a" ]; then
    AIM_ARGS="--aim --aim-experiment ablation-study"
    echo "Aim tracking enabled"
fi

# Run in background with proper output redirection
nohup bash -c "
    echo '=== Ablation Study Started ===' > logs/ablation_status.txt
    echo 'Date: \$(date)' >> logs/ablation_status.txt
    echo 'PID: \$\$' >> logs/ablation_status.txt
    echo 'Aim: ${AIM_ARGS}' >> logs/ablation_status.txt
    echo '' >> logs/ablation_status.txt
    
    uv run python -u scripts/ablation_loss_study.py \
        --data data/word_frequency.csv \
        --epochs 15 \
        --batch-size 64 \
        --output ablation_results.json \
        ${AIM_ARGS} \
        > logs/ablation_output.log 2>&1
    
    echo '' >> logs/ablation_status.txt
    echo '=== Ablation Study Completed ===' >> logs/ablation_status.txt
    echo 'Date: \$(date)' >> logs/ablation_status.txt
    echo 'Exit code: \$?' >> logs/ablation_status.txt
" > logs/ablation_nohup.log 2>&1 &

ABLATION_PID=$!
echo $ABLATION_PID > logs/ablation.pid

echo "Ablation study started in background"
echo "PID: $ABLATION_PID"
echo "Status: logs/ablation_status.txt"
echo "Output: logs/ablation_output.log"
if [ -n "$AIM_ARGS" ]; then
    echo "Aim: Enabled (experiment: ablation-study)"
fi
echo "Monitor: tail -f logs/ablation_output.log"

