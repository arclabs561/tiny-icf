#!/bin/bash
# Download training data for tiny-icf

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

mkdir -p "$DATA_DIR"

echo "📥 Downloading training data for tiny-icf..."
echo ""

# Build best available frequency lists (wordfreq-based).
if [ ! -f "$DATA_DIR/word_frequency.csv" ]; then
    echo "Building best frequency lists via wordfreq..."
    cd "$PROJECT_ROOT"
    uv run python scripts/download_best_data.py
else
    echo "✓ word_frequency.csv already exists"
    echo "  (If you want to rebuild from wordfreq, run: uv run python scripts/download_best_data.py)"
fi

echo ""
echo "✅ Data download complete!"
echo "   Data directory: $DATA_DIR"

