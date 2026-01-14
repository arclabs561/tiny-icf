#!/bin/bash
# Launch on-demand instance and automatically setup training
# Usage: ./scripts/launch_ondemand_training.sh [instance-type] [auto-start]

set -e

INSTANCE_TYPE="${1:-m5.4xlarge}"
AUTO_START="${2:-true}"
REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${AWS_KEY_NAME:-tarek}"

echo "🚀 Launching on-demand instance for training"
echo "   Type: $INSTANCE_TYPE"
echo "   Auto-start: $AUTO_START"
echo ""

# Launch instance
INSTANCE_ID=$(./scripts/scale_gpu_training.sh on-demand "$INSTANCE_TYPE" 2>&1 | grep -E "Instance ready|Instance launching" | grep -oE "i-[a-z0-9]+" | head -1)

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ Failed to launch instance"
    exit 1
fi

echo "✅ Instance: $INSTANCE_ID"
echo "⏳ Waiting for instance to be ready for SSH..."

# Wait for SSH
sleep 30
for i in {1..12}; do
    if ssh -i ~/.ssh/${KEY_NAME}.pem -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null) "echo 'SSH ready'" 2>/dev/null; then
        echo "✅ SSH ready"
        break
    fi
    [ $i -eq 12 ] && { echo "⚠️  SSH timeout, but continuing..."; }
    sleep 10
done

# Upload and optionally start
echo ""
echo "📤 Uploading project..."
./scripts/scale_gpu_training.sh upload "$INSTANCE_ID" "$AUTO_START"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)

echo ""
echo "✅ Setup complete!"
echo ""
echo "Instance: $INSTANCE_ID"
echo "IP: $PUBLIC_IP"
echo ""
if [ "$AUTO_START" = "true" ]; then
    echo "Training started automatically. Monitor with:"
else
    echo "Start training with:"
fi
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP 'cd ~/idf-est && tail -f training.log'"

