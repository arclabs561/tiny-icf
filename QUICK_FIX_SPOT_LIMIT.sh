#!/bin/bash
# Quick fix for MaxSpotInstanceCountExceeded error

set -e

echo "🔧 Fixing Spot Instance Limit Issue"
echo "===================================="
echo ""

# Step 1: Clean up any existing resources
echo "Step 1: Cleaning up existing resources..."
./scripts/cleanup_aws_resources.sh --force 2>&1 | grep -E "✅|❌|Found|Terminating|Cancelling" || true
echo ""

# Step 2: Wait a moment for AWS to process
echo "Step 2: Waiting for AWS to process cleanup (10 seconds)..."
sleep 10
echo ""

# Step 3: Try different regions
REGIONS=("us-east-1" "us-west-2" "us-west-1" "eu-west-1")

for REGION in "${REGIONS[@]}"; do
    echo "Step 3: Trying region: $REGION"
    export AWS_REGION=$REGION
    
    # Quick test - just check if we can make a request
    TEST_OUTPUT=$(aws ec2 request-spot-instances \
        --instance-count 1 \
        --launch-specification '{"ImageId":"ami-03deb8c961063af8c","InstanceType":"g4dn.xlarge","KeyName":"tarek","SecurityGroupIds":["sg-de86b4ac"]}' \
        --spot-price "0.50" \
        --type "one-time" \
        --region "$REGION" \
        --output json 2>&1 || echo "ERROR")
    
    if echo "$TEST_OUTPUT" | grep -q "SpotInstanceRequestId"; then
        SPOT_REQ=$(echo "$TEST_OUTPUT" | jq -r '.SpotInstanceRequests[0].SpotInstanceRequestId' 2>/dev/null)
        echo "✅ Success! Region $REGION works. Spot Request: $SPOT_REQ"
        echo ""
        echo "Cancel test request:"
        echo "  aws ec2 cancel-spot-instance-requests --spot-instance-request-ids $SPOT_REQ --region $REGION"
        echo ""
        echo "Launch with this region:"
        echo "  AWS_REGION=$REGION ./scripts/scale_gpu_training.sh up g4dn.xlarge 24"
        exit 0
    elif echo "$TEST_OUTPUT" | grep -q "MaxSpotInstanceCountExceeded"; then
        echo "❌ Region $REGION also has limit issue"
    else
        echo "⚠️  Region $REGION: $(echo "$TEST_OUTPUT" | head -3)"
    fi
    echo ""
done

echo "❌ All regions hit limits. Options:"
echo ""
echo "1. Request limit increase from AWS Support:"
echo "   https://console.aws.amazon.com/support/ -> Create case -> Service limit increase"
echo ""
echo "2. Wait 15-30 minutes for limits to reset"
echo ""
echo "3. Use On-Demand instances instead (more expensive):"
echo "   Modify scripts/scale_gpu_training.sh to use regular EC2 instances"

