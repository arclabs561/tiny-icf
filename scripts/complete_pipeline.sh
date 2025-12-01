#!/bin/bash
# Complete pipeline: train -> validate -> export

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

source .venv/bin/activate

echo "=" | head -c 80; echo
echo "Complete Training Pipeline"
echo "=" | head -c 80; echo

# Configuration
DATA_FILE="data/multilingual/multilingual_combined.csv"
TYPO_CORPUS="data/typos/github_typos.csv"
EMOJI_FREQ="data/emojis/emoji_frequencies.csv"
MODEL_OUTPUT="models/model_curriculum_final.pt"
EPOCHS=30

# Step 1: Training
echo ""
echo "Step 1: Training Model"
echo "-" | head -c 80; echo
python -m tiny_icf.train_curriculum \
    --data "$DATA_FILE" \
    --typo-corpus "$TYPO_CORPUS" \
    --emoji-freq "$EMOJI_FREQ" \
    --multilingual \
    --include-symbols \
    --include-emojis \
    --epochs "$EPOCHS" \
    --batch-size 128 \
    --lr 0.001 \
    --curriculum-stages 5 \
    --warmup-epochs 3 \
    --augment-prob 0.2 \
    --output "$MODEL_OUTPUT" \
    --device auto

if [ ! -f "$MODEL_OUTPUT" ]; then
    echo "ERROR: Model file not created!"
    exit 1
fi

echo ""
echo "✓ Training complete: $MODEL_OUTPUT"

# Step 2: Validation
echo ""
echo "Step 2: Validating Model"
echo "-" | head -c 80; echo
python scripts/validate_trained_model.py \
    --model "$MODEL_OUTPUT" \
    --data "$DATA_FILE" \
    --device auto

# Step 3: Export weights for Rust
echo ""
echo "Step 3: Exporting Weights for Rust"
echo "-" | head -c 80; echo
python -m tiny_icf.export_nano_weights \
    "$MODEL_OUTPUT" \
    "rust/weights.json" \
    "rust/weights.bin"

if [ -f "rust/weights.json" ]; then
    echo "✓ Weights exported: rust/weights.json"
else
    echo "⚠ Weight export may have failed"
fi

# Step 4: End-to-end inference test
echo ""
echo "Step 4: End-to-End Inference Test"
echo "-" | head -c 80; echo
python -m tiny_icf.predict \
    --model "$MODEL_OUTPUT" \
    --words "the apple xylophone qzxbjk café привет 😀"

echo ""
echo "=" | head -c 80; echo
echo "Pipeline Complete!"
echo "=" | head -c 80; echo

