# Experiment History & Results

## Overview

This document tracks all experiments run on the tiny-icf model, their configurations, results, and learnings.

## Experiment Ranking (by Best Validation Spearman)

| Rank | Experiment | Best Val Spearman | Epochs | Status | Model Path |
|------|------------|-------------------|--------|--------|------------|
| 1 | Temporal AMOO | 0.1335 | 100 | ✅ Complete | `models/model_temporal_amoo.pt` |
| 2 | Aggressive Regularization | 0.1165 | 100 | ✅ Complete | `models/model_aggressive_reg.pt` |
| 3 | ResidualICF | 0.1111 | 63+ | ✅ Complete | `models/model_residual.pt` |
| 4 | BatchNorm | 0.0728 | 100 | ✅ Complete | `models/model_batchnorm.pt` |
| 5 | Reduced Capacity | 0.0482 | 100 | ✅ Complete | `models/model_reduced_capacity.pt` |

## Experiment Details

### 1. Temporal AMOO (Best: 0.1335)
**Configuration:**
- Model: UniversalICF
- Loss: Aligned Multi-Objective Optimization (AMOO) + Temporal Loss
- Temporal data: Historical n-grams (1800-2019)
- Regularization: dropout=0.4, weight_decay=1e-4
- Training: 100 epochs, batch_size=256, lr=1e-3

**Key Findings:**
- Temporal consistency loss helps with generalization
- AMOO adaptively weights multiple objectives
- Best performing experiment overall

**Script:** `scripts/train_temporal_amoo.py`

### 2. Aggressive Regularization (0.1165)
**Configuration:**
- Model: UniversalICF
- Regularization: dropout=0.5, weight_decay=1e-3, augment_prob=0.3
- Loss: CombinedLoss (Huber + Ranking, rank_weight=5.0)
- Training: 100 epochs, batch_size=256, lr=1e-3

**Key Findings:**
- Strong regularization reduces overfitting
- Higher dropout (0.5) helps generalization
- Second best performance

**Script:** `scripts/train_aggressive_regularization.py`

### 3. ResidualICF (0.1111)
**Configuration:**
- Model: ResidualICF (residual connections + BatchNorm)
- Parameters: 30,943
- Regularization: dropout=0.4, weight_decay=1e-4
- Loss: CombinedLoss (Huber + Ranking, rank_weight=5.0)
- Training: 100 epochs, batch_size=256, lr=1e-3

**Key Findings:**
- Residual connections improve gradient flow
- 54% improvement over BatchNorm alone
- Good balance of capacity and performance

**Script:** `scripts/train_residual.py`

### 4. BatchNorm (0.0728)
**Configuration:**
- Model: UniversalICF with BatchNorm layers
- Parameters: 25,003
- Regularization: dropout=0.4, weight_decay=1e-4
- Loss: CombinedLoss (Huber + Ranking, rank_weight=5.0)
- Training: 100 epochs, batch_size=256, lr=1e-3

**Key Findings:**
- BatchNorm helps but not as much as residuals
- Normalization reduces internal covariate shift
- Baseline for architectural improvements

**Script:** `scripts/train_batchnorm.py`

### 5. Reduced Capacity (0.0482)
**Configuration:**
- Model: UniversalICF (reduced dimensions)
- Parameters: 24,895 (37.9% reduction)
- Config: emb=36, conv=18, hidden=36
- Regularization: dropout=0.4, weight_decay=1e-4
- Loss: CombinedLoss (Huber + Ranking, rank_weight=5.0)
- Training: 100 epochs, batch_size=256, lr=1e-3

**Key Findings:**
- Reducing capacity too much hurts performance
- Model needs sufficient capacity to learn patterns
- Overfitting not solved by just reducing size

**Script:** `scripts/train_reduced_capacity.py`

## Other Experiments

### Gated ResidualICF
- Model: GatedResidualICF (learnable gates)
- Status: Ready to train
- Hypothesis: Gated residuals may outperform simple residuals

**Script:** `scripts/train_gated_residual.py`

### NanoICF
- Model: NanoICF (ultra-small, 6,721 params)
- Status: Ready to train
- Hypothesis: Minimal model for edge deployment

**Script:** `scripts/train_nano.py`

### Loss Function Experiments
- **rank_weight=5.0**: Best Spearman 0.1733 (training), 0.2842 (full dataset)
- **rank_weight=10.0**: Best Spearman 0.1736 (training), 0.2354 (full dataset), 80% Jabberwocky

**Key Finding:** rank_weight=5.0 better for overall correlation, rank_weight=10.0 better for rare word detection.

## Common Patterns

### What Works
1. ✅ Residual connections (54% improvement)
2. ✅ Strong regularization (dropout=0.5, weight_decay=1e-3)
3. ✅ Temporal consistency (AMOO experiment)
4. ✅ BatchNorm for normalization
5. ✅ Combined loss (Huber + Ranking)

### What Doesn't Work
1. ❌ Reducing capacity too much (hurts performance)
2. ❌ Too high ranking weight (hurts absolute accuracy)
3. ❌ Training too short (need 100+ epochs)

## Running Experiments

### Quick Start
```bash
# Run comparison of all experiments
python scripts/compare_all_experiments.py

# Run specific experiment
python scripts/train_residual.py \
    --data data/word_frequency.csv \
    --output-dir models \
    --epochs 100
```

### Ephemeral Training (RunPod)
```bash
# Setup and start training
./scripts/setup_ephemeral_training.sh
./scripts/run_ephemeral_training.sh start

# Monitor
./scripts/monitor_ephemeral.sh
```

## Next Experiments to Try

1. **Ensemble Models**: Combine best models
2. **Longer Training**: 200+ epochs for temporal AMOO
3. **Data Augmentation**: More aggressive augmentation
4. **Learning Rate Schedules**: Cosine annealing, warm restarts
5. **Multi-Task Learning**: ICF + language detection + temporal prediction

