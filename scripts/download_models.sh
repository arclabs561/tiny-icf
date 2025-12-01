#!/bin/bash
# Download pre-trained models for tiny-icf

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

mkdir -p "$MODELS_DIR"

echo "📥 Downloading pre-trained models for tiny-icf..."
echo ""

# Check if models already exist
if [ -f "$MODELS_DIR/model_local_v3.pt" ]; then
    echo "✓ Models already exist in $MODELS_DIR"
    echo ""
    ls -lh "$MODELS_DIR"/*.pt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
    echo "To download fresh models, remove existing ones first:"
    echo "  rm -rf $MODELS_DIR/*.pt"
    exit 0
fi

echo "⚠️  Pre-trained models not available for download yet."
echo ""
echo "To train your own model:"
echo "  python -m tiny_icf.train --data data/word_frequency.csv --epochs 50 --output models/model.pt"
echo ""
echo "Or use the curriculum training:"
echo "  python -m tiny_icf.train_curriculum --data data/word_frequency.csv --epochs 50 --output models/model.pt"
echo ""

