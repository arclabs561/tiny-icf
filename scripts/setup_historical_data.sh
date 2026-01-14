#!/bin/bash
# Setup script for downloading historical n-gram data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/historical_ngrams"

echo "=== Historical N-gram Data Setup ==="
echo "Data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

# Check if Python dependencies are available
if ! python3 -c "import pandas, numpy, tqdm" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install pandas numpy tqdm
fi

echo ""
echo "Options:"
echo "1. Download 1-gram data for key years (1800, 1900, 2000) - ~500MB each"
echo "2. Download full decade data (1800-2010) - ~5GB total"
echo "3. Process existing files only"
echo ""
read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo "Downloading 1-gram data for 1800, 1900, 2000..."
        cd "$PROJECT_ROOT"
        uv run --python 3.12 scripts/download_historical_ngrams.py \
            --output-dir "$DATA_DIR" \
            --years 1800 1900 2000 \
            --ngram-type 1gram \
            --min-count 5
        ;;
    2)
        echo "Downloading full decade data (this will take a while)..."
        cd "$PROJECT_ROOT"
        years=$(seq 1800 10 2010 | tr '\n' ' ')
        uv run --python 3.12 scripts/download_historical_ngrams.py \
            --output-dir "$DATA_DIR" \
            --years $years \
            --ngram-type 1gram \
            --min-count 5
        ;;
    3)
        echo "Processing existing files..."
        cd "$PROJECT_ROOT"
        uv run --python 3.12 scripts/download_historical_ngrams.py \
            --output-dir "$DATA_DIR" \
            --process-only \
            --ngram-type 1gram \
            --min-count 5
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=== Setup Complete ==="
echo "Historical data available at: $DATA_DIR"

