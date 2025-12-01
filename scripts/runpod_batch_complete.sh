#!/bin/bash
# Complete non-interactive batch training workflow for RunPod
# Usage: ./runpod_batch_complete.sh [gpu-type]

set -e

GPU_TYPE="${1:-NVIDIA GeForce RTX 4080 SUPER}"
POD_NAME="tiny-icf-batch-$(date +%s)"

echo "🚀 RunPod Batch Training Workflow"
echo "=================================="
echo "GPU: $GPU_TYPE"
echo "Pod: $POD_NAME"
echo ""

# Get API key from MCP config
API_KEY=$(grep -o '"RUNPOD_API_KEY": "[^"]*"' ~/.cursor/mcp.json | cut -d'"' -f4)

if [ -z "$API_KEY" ]; then
    echo "❌ Could not find RUNPOD_API_KEY in ~/.cursor/mcp.json"
    exit 1
fi

# Configure runpodctl
echo "📝 Configuring runpodctl..."
runpodctl config --apiKey "$API_KEY" >/dev/null 2>&1

# Create pod
echo "📦 Creating pod..."
POD_OUTPUT=$(runpodctl create pod \
    --name "$POD_NAME" \
    --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
    --gpuType "$GPU_TYPE" \
    --containerDiskSize 30 \
    --mem 32 \
    --env "PYTHONUNBUFFERED=1" 2>&1)

POD_ID=$(echo "$POD_OUTPUT" | grep -oE '[a-z0-9]{13,}' | head -1)

if [ -z "$POD_ID" ]; then
    echo "❌ Failed to create pod"
    echo "$POD_OUTPUT"
    exit 1
fi

echo "✅ Pod created: $POD_ID"
echo "   Waiting 30 seconds for pod to be ready..."
sleep 30

# Upload project
echo "📤 Uploading project..."
runpodctl send . "$POD_ID" /workspace/tiny-icf

# Start training (non-interactive, runs in background)
echo "🚀 Starting batch training..."
echo "   Training will run automatically and exit when complete"
echo ""

# Get SSH command for monitoring
SSH_CMD=$(runpodctl ssh connect "$POD_ID" 2>&1 | grep -oE 'ssh [^ ]+@[^ ]+' || echo "")

if [ -n "$SSH_CMD" ]; then
    echo "📊 To monitor training:"
    echo "   $SSH_CMD"
    echo "   Then: tail -f /workspace/tiny-icf/training.log"
    echo ""
fi

# Run training via exec (non-interactive, PEP 723 script)
runpodctl exec "$POD_ID" -- bash -c "
cd /workspace/tiny-icf
nohup uv run scripts/train_batch.py \\
    --data data/multilingual/multilingual_combined.csv \\
    --typo-corpus data/typos/github_typos.csv \\
    --output-dir /workspace/models \\
    --epochs 50 \\
    --batch-size 256 \\
    --devices 1 \\
    > training.log 2>&1 &
echo \$! > training.pid
echo 'Training started with PID: \$(cat training.pid)'
"

echo ""
echo "✅ Training started in background"
echo ""
echo "📋 Next steps:"
echo "   1. Monitor: runpodctl ssh connect $POD_ID"
echo "   2. Download results: runpodctl receive $POD_ID /workspace/models ./models"
echo "   3. Stop pod: runpodctl stop pod $POD_ID"
echo ""
echo "   Or use: python3 scripts/runpod_batch_training.py --download-only --pod-id $POD_ID"

