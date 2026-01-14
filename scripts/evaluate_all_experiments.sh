#!/bin/bash
# Evaluate all three experiments and create comparison report

set -e

echo "=" | cat
echo "Evaluating All Experiments"
echo "=" | cat
echo ""

DATA="data/word_frequency.csv"
OUTPUT_DIR="evaluation_results"
mkdir -p "$OUTPUT_DIR"

# Function to evaluate a model
evaluate_model() {
    local model_path=$1
    local output_file=$2
    local name=$3
    
    echo "Evaluating $name..."
    uv run --python 3.12 scripts/evaluate_model.py \
        --model "$model_path" \
        --data "$DATA" \
        > "$output_file" 2>&1
    
    echo "  Results saved to: $output_file"
    echo ""
}

# Evaluate each model
if [ -f "models/model_diagnostic_rank5.pt" ]; then
    evaluate_model \
        "models/model_diagnostic_rank5.pt" \
        "$OUTPUT_DIR/rank5_evaluation.txt" \
        "rank_weight=5.0"
fi

if [ -f "models/model_diagnostic_rank10.pt" ]; then
    evaluate_model \
        "models/model_diagnostic_rank10.pt" \
        "$OUTPUT_DIR/rank10_evaluation.txt" \
        "rank_weight=10.0"
fi

if [ -f "models/model_calibrated.pt" ]; then
    evaluate_model \
        "models/model_calibrated.pt" \
        "$OUTPUT_DIR/calibrated_evaluation.txt" \
        "calibrated"
fi

# Create summary
echo "=" | cat
echo "Creating Summary"
echo "=" | cat

python3 << 'PYTHON_EOF'
import re
from pathlib import Path

output_dir = Path("evaluation_results")
results = {}

for file in output_dir.glob("*_evaluation.txt"):
    name = file.stem.replace("_evaluation", "")
    with open(file) as f:
        content = f.read()
    
    # Extract metrics
    spearman_match = re.search(r'Spearman:\s+([\d.]+)', content)
    mae_match = re.search(r'MAE:\s+([\d.]+)', content)
    jabberwocky_match = re.search(r'Jabberwocky.*?(\d+\.?\d*)%', content)
    pred_mean_match = re.search(r'Predictions:.*?mean=([\d.]+)', content)
    
    results[name] = {
        'spearman': float(spearman_match.group(1)) if spearman_match else None,
        'mae': float(mae_match.group(1)) if mae_match else None,
        'jabberwocky': float(jabberwocky_match.group(1)) if jabberwocky_match else None,
        'pred_mean': float(pred_mean_match.group(1)) if pred_mean_match else None,
    }

print("\nComparison Summary:")
print("=" * 70)
print(f"{'Model':<20} {'Spearman':<12} {'MAE':<10} {'Jabberwocky':<15} {'Pred Mean':<12}")
print("-" * 70)

for name, metrics in sorted(results.items()):
    spearman = f"{metrics['spearman']:.4f}" if metrics['spearman'] else "N/A"
    mae = f"{metrics['mae']:.4f}" if metrics['mae'] else "N/A"
    jabberwocky = f"{metrics['jabberwocky']:.1f}%" if metrics['jabberwocky'] else "N/A"
    pred_mean = f"{metrics['pred_mean']:.4f}" if metrics['pred_mean'] else "N/A"
    
    print(f"{name:<20} {spearman:<12} {mae:<10} {jabberwocky:<15} {pred_mean:<12}")

# Find winners
if len(results) >= 2:
    print("\n" + "=" * 70)
    print("Winners:")
    print("=" * 70)
    
    best_spearman = max(results.items(), key=lambda x: x[1]['spearman'] if x[1]['spearman'] else -1)
    best_mae = min(results.items(), key=lambda x: x[1]['mae'] if x[1]['mae'] else 999)
    best_jabberwocky = max(results.items(), key=lambda x: x[1]['jabberwocky'] if x[1]['jabberwocky'] else -1)
    
    print(f"Best Spearman: {best_spearman[0]} ({best_spearman[1]['spearman']:.4f})")
    print(f"Best MAE: {best_mae[0]} ({best_mae[1]['mae']:.4f})")
    print(f"Best Jabberwocky: {best_jabberwocky[0]} ({best_jabberwocky[1]['jabberwocky']:.1f}%)")

PYTHON_EOF

echo ""
echo "=" | cat
echo "Evaluation Complete"
echo "=" | cat
echo "Results saved to: $OUTPUT_DIR/"

