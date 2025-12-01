#!/bin/bash
# Complete post-training validation pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

source .venv/bin/activate

MODEL="models/model_curriculum_final.pt"
DATA="data/multilingual/multilingual_combined.csv"

echo "=" | head -c 80; echo
echo "Post-Training Validation Pipeline"
echo "=" | head -c 80; echo

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found: $MODEL"
    exit 1
fi

# Step 1: Full Validation
echo ""
echo "Step 1: Full Model Validation"
echo "-" | head -c 80; echo
python scripts/validate_trained_model.py \
    --model "$MODEL" \
    --data "$DATA" \
    --device auto

# Step 2: Sample Predictions
echo ""
echo "Step 2: Sample Predictions"
echo "-" | head -c 80; echo
python -m tiny_icf.predict \
    --model "$MODEL" \
    --words "the apple xylophone qzxbjk café привет 😀"

# Step 3: Performance Benchmark
echo ""
echo "Step 3: Performance Benchmark"
echo "-" | head -c 80; echo
python -c "
from tiny_icf.model import UniversalICF
import torch
import time

model = UniversalICF()
model.load_state_dict(torch.load('$MODEL', map_location='cpu'))
model.eval()

# Test inference speed
test_words = ['the', 'apple', 'xylophone', 'qzxbjk', 'café', 'привет', '😀'] * 100
start = time.time()
with torch.no_grad():
    for word in test_words:
        byte_seq = word.encode('utf-8')[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
        _ = model(tensor)
elapsed = time.time() - start
avg_time = (elapsed / len(test_words)) * 1000

print(f'Inference speed: {avg_time:.3f} ms/word')
print(f'Throughput: {1000/avg_time:.1f} words/sec')
print(f'Total time for {len(test_words)} words: {elapsed:.2f}s')
"

# Step 4: Export Weights (if not already done)
echo ""
echo "Step 4: Export Weights for Rust"
echo "-" | head -c 80; echo
if [ ! -f "rust/weights.json" ]; then
    python -m tiny_icf.export_weights \
        --model "$MODEL" \
        --json rust/weights.json \
        --bin rust/weights.bin
else
    echo "✓ Weights already exported"
fi

echo ""
echo "=" | head -c 80; echo
echo "Validation Complete!"
echo "=" | head -c 80; echo

