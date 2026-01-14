#!/bin/bash
# Auto-launch iteration 2 experiments when conditions are met

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check if any experiment has reached epoch 15+
MIN_EPOCH=15

check_ready() {
    python3 << PYTHON
import pandas as pd
from pathlib import Path

exps = ['research_aligned_residual', 'research_aligned_neural_sort', 
        'research_aligned_strong_reg', 'research_aligned_high_spearman']
max_epoch = 0

for exp in exps:
    mf = Path(f'models/{exp}/lightning_logs/version_0/metrics.csv')
    if mf.exists():
        df = pd.read_csv(mf)
        val_col = [c for c in df.columns if 'val' in c.lower() and 'spearman' in c.lower()]
        if val_col:
            val_df = df[df[val_col[0]].notna()]
            if len(val_df) > 0:
                epoch = int(val_df['epoch'].iloc[-1])
                max_epoch = max(max_epoch, epoch)

print(max_epoch)
PYTHON
}

MAX_EPOCH=$(check_ready)

if [ "$MAX_EPOCH" -ge "$MIN_EPOCH" ]; then
    echo "✅ Conditions met (max epoch: $MAX_EPOCH >= $MIN_EPOCH)"
    echo "🚀 Launching iteration 2 experiments..."
    
    for exp in research_aligned_monotonicity research_aligned_high_lr research_aligned_probabilistic; do
        if [ ! -f "models/${exp}/lightning_logs/version_0/metrics.csv" ]; then
            echo "   Launching: $exp"
            mkdir -p "models/${exp}"
            nohup uv run python ../trainctl/training/scripts/train_flexible_opportunistic.py \
                --data data/word_frequency.csv \
                --experiments "$exp" \
                --max_experiments 1 \
                > "models/${exp}/training.log" 2>&1 &
            sleep 2
        else
            echo "   ⏭️  $exp already running"
        fi
    done
    
    echo "✅ Iteration 2 experiments launched!"
else
    echo "⏳ Not ready yet (max epoch: $MAX_EPOCH < $MIN_EPOCH)"
    echo "   Waiting for experiments to reach epoch $MIN_EPOCH..."
fi
