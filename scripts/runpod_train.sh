#!/bin/bash
# Complete RunPod training workflow using API key from MCP config

set -e

# Get API key from MCP config
API_KEY=$(grep -o '"RUNPOD_API_KEY": "[^"]*"' ~/.cursor/mcp.json | cut -d'"' -f4)

if [ -z "$API_KEY" ]; then
    echo "❌ Failed to get API key from MCP config"
    exit 1
fi

# Configure runpodctl
runpodctl config --apiKey "$API_KEY" >/dev/null 2>&1

# Get pod ID (use first running pod or create new one)
POD_ID=$(runpodctl get pod 2>&1 | grep RUNNING | head -1 | awk '{print $1}')

if [ -z "$POD_ID" ]; then
    echo "No running pod found. Creating new pod..."
    POD_NAME="tiny-icf-training-$(date +%s)"
    OUTPUT=$(runpodctl create pod \
        --name "$POD_NAME" \
        --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
        --gpuType "NVIDIA GeForce RTX 3090" \
        --volumeSize 50 \
        --containerDiskSize 20 \
        --startSSH \
        --env "PYTHONUNBUFFERED=1" 2>&1)
    
    POD_ID=$(echo "$OUTPUT" | grep -oE '[a-z0-9]{13}' | head -1)
    echo "✅ Pod created: $POD_ID"
    echo "Waiting 30 seconds for pod to be ready..."
    sleep 30
else
    echo "✅ Using existing pod: $POD_ID"
fi

echo ""
echo "📤 Uploading project files..."
runpodctl send "$POD_ID" . /workspace/tiny-icf

echo ""
echo "🚀 Starting training..."
echo "Pod ID: $POD_ID"
echo ""
echo "To monitor training:"
echo "  runpodctl exec $POD_ID -- tail -f /workspace/tiny-icf/training.log"
echo ""
echo "To SSH into pod:"
echo "  runpodctl ssh $POD_ID"
echo ""

# Start training in background
runpodctl exec "$POD_ID" -- bash -c "
cd /workspace/tiny-icf
uv pip install -e . 2>&1 | tee install.log
python -m tiny_icf.train_curriculum \
    --data data/multilingual/multilingual_combined.csv \
    --typo-corpus data/typos/github_typos.csv \
    --multilingual \
    --epochs 50 \
    --batch-size 128 \
    --output models/model_runpod.pt \
    2>&1 | tee training.log
" &

echo "Training started in background. Use the commands above to monitor progress."

