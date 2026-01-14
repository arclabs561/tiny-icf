#!/bin/bash
# Safe cleanup of AWS spot instances
# Usage: ./scripts/cleanup_aws_resources.sh [--force] [--dry-run]

set -e

DRY_RUN=false
FORCE=false
REGION="${AWS_REGION:-us-east-1}"

[ "$1" = "--dry-run" ] && DRY_RUN=true
[ "$1" = "--force" ] && FORCE=true

echo "🧹 AWS Resource Cleanup ($([ "$DRY_RUN" = true ] && echo "DRY RUN" || echo "LIVE"))"

# Find running spot instances
SPOT_INSTANCES=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=instance-state-name,Values=running,pending" "Name=instance-lifecycle,Values=spot" \
    --query 'Reservations[*].Instances[*].[InstanceId,InstanceType]' \
    --output text 2>/dev/null || echo "")

if [ -n "$SPOT_INSTANCES" ] && [ "$SPOT_INSTANCES" != "None" ]; then
    echo "$SPOT_INSTANCES" | while read -r INSTANCE_ID INSTANCE_TYPE; do
        [ -z "$INSTANCE_ID" ] && continue
        echo "  $INSTANCE_ID ($INSTANCE_TYPE)"
        [ "$DRY_RUN" = false ] && [ "$FORCE" = true ] && \
            aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null 2>&1 && \
            echo "    ✅ Terminated"
    done
else
    echo "✅ No running spot instances"
fi

# Find active spot requests
SPOT_REQUESTS=$(aws ec2 describe-spot-instance-requests \
    --region "$REGION" \
    --filters "Name=state,Values=open,active" \
    --query 'SpotInstanceRequests[*].[SpotInstanceRequestId,State]' \
    --output text 2>/dev/null || echo "")

if [ -n "$SPOT_REQUESTS" ] && [ "$SPOT_REQUESTS" != "None" ]; then
    echo "$SPOT_REQUESTS" | while read -r REQUEST_ID STATE; do
        [ -z "$REQUEST_ID" ] && continue
        echo "  Request: $REQUEST_ID ($STATE)"
        [ "$DRY_RUN" = false ] && [ "$FORCE" = true ] && \
            aws ec2 cancel-spot-instance-requests --spot-instance-request-ids "$REQUEST_ID" --region "$REGION" >/dev/null 2>&1 && \
            echo "    ✅ Cancelled"
    done
else
    echo "✅ No active spot requests"
fi

[ "$DRY_RUN" = true ] && echo "🔍 Dry run - no changes made"
