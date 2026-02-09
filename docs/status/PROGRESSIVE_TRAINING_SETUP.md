# Progressive Training Setup

## Overview
Progressive training runs experiments in order of increasing complexity:
1. **Stage 1**: Simple baseline (50 epochs)
2. **Stage 2**: Moderate complexity (100 epochs)  
3. **Stage 3**: Full complexity (150 epochs)

## Experiments

### Stage 1: Baseline
- **Model**: Universal
- **Features**: NeuralNDCG only
- **Batch**: 64
- **LR**: 1e-3
- **Epochs**: 50

### Stage 2: Moderate
- **Models**: Universal + Residual
- **Features**: NeuralNDCG + LambdaRank
- **Batch**: 64
- **LR**: 1e-3
- **Epochs**: 100

### Stage 3: Full Complexity
- **Models**: All (Universal + Residual)
- **Features**: NeuralNDCG + LambdaRank + Aggressive Regularization
- **Batch**: 512 (aggressive)
- **LR**: 5e-4
- **Weight Decay**: 1e-3
- **Epochs**: 150

## Usage

### On GPU Instance

```bash
# Upload project
./scripts/scale_gpu_training.sh upload <instance-id>

# SSH to instance
ssh -i ~/.ssh/tarek.pem ubuntu@<public-ip>

# Run progressive training
cd ~/idf-est
./scripts/train_progressive_experiments.sh data/word_frequency.csv models/progressive
```

### Or run individual stages

```bash
# Stage 1 only
uv run scripts/train_flexible_opportunistic.py \
    --data data/word_frequency.csv \
    --max_experiments 1 \
    --experiments standard_enhanced \
    --train_split 0.8

# Stage 2
uv run scripts/train_flexible_opportunistic.py \
    --data data/word_frequency.csv \
    --max_experiments 2 \
    --experiments standard_enhanced residual_listwise \
    --train_split 0.8

# Stage 3 (all experiments)
uv run scripts/train_flexible_opportunistic.py \
    --data data/word_frequency.csv \
    --max_experiments 3 \
    --train_split 0.8
```

## Expected Results

Each stage produces:
- Model checkpoints in `models/`
- Training logs in `models/progressive/stage*_*.log`
- Loss curves and metrics
- Best model saved per experiment

## Monitoring

Check training progress:
```bash
tail -f models/progressive/stage1_baseline.log
tail -f models/progressive/stage2_moderate.log
tail -f models/progressive/stage3_full.log
```

