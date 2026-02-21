#!/bin/bash
# Upload a trained model to S3.
#
# Requires: aws CLI, credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or ~/.aws/credentials)
#
# Usage:
#   ./scripts/upload_model_to_s3.sh models/multitask_all_fronts_v3.pt s3://your-bucket/tiny-icf/
#   BUCKET=my-bucket ./scripts/upload_model_to_s3.sh models/multitask_en.pt

set -e

MODEL_PATH="${1:?Usage: $0 <model.pt> [s3://bucket/prefix/]}"
S3_DEST="${2:-}"

if [ -z "$S3_DEST" ]; then
  BUCKET="${BUCKET:-}"
  if [ -z "$BUCKET" ]; then
    echo "Error: Pass s3://bucket/prefix as second arg, or set BUCKET=your-bucket"
    exit 1
  fi
  S3_DEST="s3://${BUCKET}/tiny-icf/"
fi

# Ensure trailing slash for prefix
case "$S3_DEST" in
  */) ;;
  *) S3_DEST="${S3_DEST}/" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model not found: $MODEL_PATH"
  exit 1
fi

BASENAME="$(basename "$MODEL_PATH")"
S3_URI="${S3_DEST}${BASENAME}"

echo "Uploading $MODEL_PATH -> $S3_URI"
aws s3 cp "$MODEL_PATH" "$S3_URI" --content-type "application/octet-stream"

echo "Done. Download with: aws s3 cp $S3_URI ."
