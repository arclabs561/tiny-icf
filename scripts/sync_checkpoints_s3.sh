#!/bin/bash
# Sync training checkpoints to/from S3
# Usage: ./scripts/sync_checkpoints_s3.sh [upload|download] <s3-bucket> [local-path]

set -e

ACTION="${1:-upload}"
S3_BUCKET="${2:-}"
LOCAL_PATH="${3:-models}"

[ -z "$S3_BUCKET" ] && { echo "Usage: $0 [upload|download] <s3-bucket> [local-path]"; exit 1; }

S3_PATH="s3://${S3_BUCKET}/checkpoints/"

case "$ACTION" in
    upload|up)
        [ ! -d "$LOCAL_PATH" ] && { echo "❌ Path not found: $LOCAL_PATH"; exit 1; }
        echo "📤 Uploading to $S3_PATH..."
        aws s3 sync "$LOCAL_PATH" "$S3_PATH" --exclude "*" --include "*.pt" --quiet
        echo "✅ Complete"
        ;;
    download|down)
        mkdir -p "$LOCAL_PATH"
        echo "📥 Downloading from $S3_PATH..."
        aws s3 sync "$S3_PATH" "$LOCAL_PATH" --exclude "*" --include "*.pt" --quiet
        echo "✅ Complete"
        ;;
    *)
        echo "Usage: $0 [upload|download] <s3-bucket> [local-path]"
        exit 1
        ;;
esac
