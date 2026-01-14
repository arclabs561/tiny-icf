#!/bin/bash
# Retry script that keeps trying until VM service is available
# Usage: ./scripts/lyceum_train_retry.sh [hardware] [epochs] [batch-size]

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

HARDWARE="${1:-a100}"
EPOCHS="${2:-100}"
BATCH_SIZE="${3:-256}"

MAX_RETRIES=20
RETRY_INTERVAL=30

echo "🔄 Retry script: Will attempt to start training every ${RETRY_INTERVAL}s"
echo "   Hardware: $HARDWARE"
echo "   Max retries: $MAX_RETRIES"
echo ""

RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    RETRY=$((RETRY + 1))
    echo "Attempt $RETRY/$MAX_RETRIES..."
    
    # Try to start VM
    if ./scripts/lyceum_train.sh "$HARDWARE" "$EPOCHS" "$BATCH_SIZE" 2>&1; then
        echo "✅ Training started successfully!"
        exit 0
    fi
    
    EXIT_CODE=$?
    
    # If it's a 503, keep retrying
    if [ $RETRY -lt $MAX_RETRIES ]; then
        echo "⏳ Service unavailable, waiting ${RETRY_INTERVAL}s before retry..."
        sleep $RETRY_INTERVAL
    else
        echo "❌ Max retries reached. Service still unavailable."
        exit 1
    fi
done




