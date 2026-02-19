#!/bin/bash
# Setup script for building historical (temporal) ICF data from Google Books

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/historical_ngrams"

echo "=== Historical N-gram Data Setup ==="
echo "Data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

echo ""
echo "Options:"
echo "1. Build temporal ICF for key decades (1800, 1900, 2000) [recommended]"
echo "2. Build temporal ICF for all decades (1800-2010, step 10) [large]"
echo ""
read -p "Choose option (1-2): " choice

case $choice in
    1)
        echo "Building temporal ICF for decades: 1800, 1900, 2000"
        cd "$PROJECT_ROOT"
        uv run --python 3.12 scripts/build_googlebooks_temporal_icf.py \
            --corpus eng \
            --release 20200217 \
            --vocab "$PROJECT_ROOT/data/word_frequency.csv" \
            --vocab-max 200000 \
            --decades 1800,1900,2000 \
            --min-count 5 \
            --cache-dir "$DATA_DIR" \
            --resume \
            --output "$DATA_DIR/historical_icf_1gram.csv"
        ;;
    2)
        echo "Building temporal ICF for decades: 1800..2010 (step 10) (this will take a while)..."
        cd "$PROJECT_ROOT"
        decades=$(seq 1800 10 2010 | tr '\n' ',' | sed 's/,$//')
        uv run --python 3.12 scripts/build_googlebooks_temporal_icf.py \
            --corpus eng \
            --release 20200217 \
            --vocab "$PROJECT_ROOT/data/word_frequency.csv" \
            --vocab-max 500000 \
            --decades "$decades" \
            --min-count 5 \
            --cache-dir "$DATA_DIR" \
            --resume \
            --output "$DATA_DIR/historical_icf_1gram.csv"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=== Setup Complete ==="
echo "Historical data available at: $DATA_DIR"
echo "Temporal ICF CSV: $DATA_DIR/historical_icf_1gram.csv"

