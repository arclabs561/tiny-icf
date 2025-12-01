#!/bin/bash
# Complete RunPod training setup script
# Uses runpodctl to create pod, upload files, and start training

set -e

echo "🚀 RunPod Training Setup"
echo "======================"

# Configuration
POD_NAME="tiny-icf-training-$(date +%s)"
IMAGE="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
GPU_TYPE="NVIDIA RTX 3090"
VOLUME_GB=50
CONTAINER_DISK_GB=20

echo "Pod Name: $POD_NAME"
echo "Image: $IMAGE"
echo "GPU: $GPU_TYPE"
echo ""

# Check if runpodctl is configured
if ! runpodctl config get >/dev/null 2>&1; then
    echo "❌ runpodctl not configured"
    echo "Run: runpodctl config --apiKey YOUR_API_KEY"
    exit 1
fi

echo "✅ runpodctl configured"

# Create pod
echo ""
echo "Creating pod..."
POD_ID=$(runpodctl create pod \
    --name "$POD_NAME" \
    --image "$IMAGE" \
    --gpu-type-id "$GPU_TYPE" \
    --volume-in-gb "$VOLUME_GB" \
    --container-disk-in-gb "$CONTAINER_DISK_GB" \
    --env "PYTHONUNBUFFERED=1" \
    --ssh-public-key "$HOME/.ssh/id_rsa.pub" 2>&1 | grep -oP 'pod-[a-zA-Z0-9]+' | head -1)

if [ -z "$POD_ID" ]; then
    echo "❌ Failed to create pod"
    exit 1
fi

echo "✅ Pod created: $POD_ID"
echo "Waiting for pod to be ready..."

# Wait for pod to be ready
sleep 10

# Upload project files
echo ""
echo "Uploading project files..."
runpodctl send "$POD_ID" . /workspace/tiny-icf

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. SSH into pod: runpodctl ssh $POD_ID"
echo "2. Install dependencies: cd /workspace/tiny-icf && uv pip install -e ."
echo "3. Run training: python -m tiny_icf.train_curriculum \\"
echo "     --data data/multilingual/multilingual_combined.csv \\"
echo "     --typo-corpus data/typos/github_typos.csv \\"
echo "     --multilingual \\"
echo "     --epochs 50 \\"
echo "     --output models/model_runpod.pt"
echo ""
echo "Or run the automated script:"
echo "  runpodctl exec $POD_ID -- bash /workspace/tiny-icf/scripts/train_on_runpod.sh"

