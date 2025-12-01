#!/bin/bash
# Download training data for tiny-icf

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

mkdir -p "$DATA_DIR"

echo "📥 Downloading training data for tiny-icf..."
echo ""

# Main word frequency data
if [ ! -f "$DATA_DIR/word_frequency.csv" ]; then
    echo "Downloading word frequency data..."
    # Add your download URL here
    # curl -L -o "$DATA_DIR/word_frequency.csv" "YOUR_URL"
    echo "⚠️  Please download word_frequency.csv and place it in data/"
    echo "   Or use: python scripts/download_datasets.py"
else
    echo "✓ word_frequency.csv already exists"
fi

# Run Python download script if available
if [ -f "$PROJECT_ROOT/scripts/download_datasets.py" ]; then
    echo ""
    echo "Running download_datasets.py..."
    cd "$PROJECT_ROOT"
    python scripts/download_datasets.py || echo "⚠️  Download script failed, check manually"
fi

echo ""
echo "✅ Data download complete!"
echo "   Data directory: $DATA_DIR"

