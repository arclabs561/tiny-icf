#!/bin/bash
# Create RunPod pod for training using API key from MCP config

set -e

# Get API key from MCP config (using grep to avoid JSON parsing issues)
API_KEY=$(grep -o '"RUNPOD_API_KEY": "[^"]*"' ~/.cursor/mcp.json | cut -d'"' -f4)

if [ -z "$API_KEY" ]; then
    echo "❌ Failed to get API key from MCP config"
    exit 1
fi

echo "🔑 Using API key from MCP config"
echo "Configuring runpodctl..."
runpodctl config --apiKey "$API_KEY"

echo ""
echo "✅ runpodctl configured"
echo ""

# Pod configuration
POD_NAME="tiny-icf-training-$(date +%s)"
IMAGE="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
GPU_TYPE="NVIDIA RTX 3090"

echo "Creating pod: $POD_NAME"
echo "Image: $IMAGE"
echo "GPU: $GPU_TYPE"
echo ""

# Create pod
echo "Creating pod (this may take a minute)..."
POD_OUTPUT=$(runpodctl create pod \
    --name "$POD_NAME" \
    --image "$IMAGE" \
    --gpu-type-id "$GPU_TYPE" \
    --volume-in-gb 50 \
    --container-disk-in-gb 20 \
    --env "PYTHONUNBUFFERED=1" 2>&1)

echo "$POD_OUTPUT"

# Extract pod ID
POD_ID=$(echo "$POD_OUTPUT" | grep -oE 'pod-[a-zA-Z0-9]+' | head -1)

if [ -z "$POD_ID" ]; then
    echo "❌ Failed to create pod or extract pod ID"
    echo "Output: $POD_OUTPUT"
    exit 1
fi

echo ""
echo "✅ Pod created: $POD_ID"
echo ""
echo "Waiting for pod to be ready (30 seconds)..."
sleep 30

echo ""
echo "📤 Uploading project files..."
runpodctl send "$POD_ID" . /workspace/tiny-icf

echo ""
echo "✅ Setup complete!"
echo ""
echo "Pod ID: $POD_ID"
echo ""
echo "Next steps:"
echo "1. SSH into pod: runpodctl ssh $POD_ID"
echo "2. Install dependencies:"
echo "   cd /workspace/tiny-icf"
echo "   uv pip install -e ."
echo "3. Start training:"
echo "   python -m tiny_icf.train_curriculum \\"
echo "     --data data/multilingual/multilingual_combined.csv \\"
echo "     --typo-corpus data/typos/github_typos.csv \\"
echo "     --multilingual \\"
echo "     --epochs 50 \\"
echo "     --batch-size 128 \\"
echo "     --output models/model_runpod.pt"
echo ""
echo "Or run automated:"
echo "  runpodctl exec $POD_ID -- bash -c 'cd /workspace/tiny-icf && uv pip install -e . && python -m tiny_icf.train_curriculum --data data/multilingual/multilingual_combined.csv --typo-corpus data/typos/github_typos.csv --multilingual --epochs 50 --batch-size 128 --output models/model_runpod.pt'"

