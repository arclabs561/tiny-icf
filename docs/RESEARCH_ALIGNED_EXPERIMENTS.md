# Research-Aligned Loss Experiments

## Overview

This document describes the research-aligned loss experiments that migrate important baselines to use the new `ResearchAlignedICFLoss` function, which incorporates:

- **Adaptive regularization strength** (matches data scale)
- **Focal loss** for hard example mining
- **Multiple ranking methods** (sigmoid, neural_sort, probabilistic, smooth_i)
- **Asymmetric penalties** (common→rare worse than rare→common)
- **Optional monotonicity constraints**
- **Optional quantile regression** for uncertainty

## Experiments

### 1. `research_aligned_standard`
**Migrated from:** `standard_improved`

**Configuration:**
- Model: UniversalICF
- Loss: ResearchAlignedICFLoss
- Spearman weight: 5.0 (matches baseline)
- Rank weight: 0.1 (matches baseline)
- Dropout: 0.25 (matches baseline)
- Adaptive regularization: ✅
- Focal loss: ✅ (gamma=2.0)
- Ranking method: sigmoid

**Purpose:** Direct comparison with the main baseline to measure improvement from research-aligned techniques.

---

### 2. `research_aligned_neural_sort`
**Advanced variant**

**Configuration:**
- Model: UniversalICF
- Loss: ResearchAlignedICFLoss
- Spearman weight: 10.0
- Rank weight: 0.5
- Ranking method: **neural_sort** (from rank-relax)
- Adaptive regularization: ✅
- Focal loss: ✅ (gamma=2.0)

**Purpose:** Test NeuralSort ranking method from rank-relax, which provides sharper gradients for ranking optimization.

---

### 3. `research_aligned_high_spearman`
**Migrated from:** `multitask_icf_high_spearman_plateau`

**Configuration:**
- Model: UniversalICF
- Loss: ResearchAlignedICFLoss
- Spearman weight: **15.0** (high focus)
- Rank weight: 0.2
- Scheduler: plateau (matches baseline)
- Adaptive regularization: ✅
- Focal loss: ✅ (gamma=2.0)

**Purpose:** Test if research-aligned loss improves high Spearman-focused training.

---

### 4. `research_aligned_strong_reg`
**Migrated from:** `multitask_icf_strong_reg`

**Configuration:**
- Model: UniversalICF
- Loss: ResearchAlignedICFLoss
- Weight decay: **2e-4** (strong regularization)
- Dropout: **0.4** (strong regularization)
- Adaptive regularization: ✅
- Focal loss: ✅ (gamma=2.0)

**Purpose:** Test research-aligned loss with strong regularization to prevent overfitting.

---

### 5. `research_aligned_residual`
**Migrated from:** `residual_listwise`

**Configuration:**
- Model: **ResidualICF** (residual connections)
- Loss: ResearchAlignedICFLoss
- Spearman weight: 10.0
- Rank weight: 0.5
- Adaptive regularization: ✅
- Focal loss: ✅ (gamma=2.0)

**Purpose:** Test research-aligned loss with ResidualICF architecture.

## Launching Experiments

### Option 1: Launch All at Once
```bash
./scripts/launch_research_aligned_experiments.sh [data_file.csv]
```

### Option 2: Launch Individual Experiments
```bash
uv run python ../trainctl/training/scripts/train_flexible_opportunistic.py \
    --data data/word_frequency.csv \
    --experiments research_aligned_standard
```

### Option 3: Launch via trainctl (if available)
```bash
trainctl aws train --experiments research_aligned_standard
```

## Monitoring

### Check Logs
```bash
tail -f models/research_aligned_*/training.log
```

### Check Metrics
```bash
cat models/research_aligned_*/lightning_logs/version_0/metrics.csv
```

### Use trainctl Monitor
```bash
trainctl monitor
```

### Compare in Aim
```bash
aim up  # Start Aim UI
# Navigate to experiment: icf-training
# Filter by run name: research_aligned_*
```

## Expected Improvements

Based on research findings, we expect:

1. **Adaptive regularization**: Better gradient flow, especially for ranking operations
2. **Focal loss**: Better handling of hard examples (words with ambiguous ICF)
3. **Asymmetric penalties**: Better handling of common→rare vs rare→common errors
4. **NeuralSort**: Sharper gradients for ranking optimization (in `research_aligned_neural_sort`)

## Comparison with Baselines

After training, compare:

- `research_aligned_standard` vs `standard_improved`
- `research_aligned_high_spearman` vs `multitask_icf_high_spearman_plateau`
- `research_aligned_strong_reg` vs `multitask_icf_strong_reg`
- `research_aligned_residual` vs `residual_listwise`

Use Aim or `scripts/compare_experiments.py` for systematic comparison.

## Next Steps

1. **Monitor training progress** for all 5 experiments
2. **Compare results** with baseline experiments
3. **Tune hyperparameters** if needed (focal_gamma, asymmetry_factor, etc.)
4. **Try advanced features**:
   - Enable monotonicity constraints (`use_monotonicity: True`)
   - Enable quantile regression (`use_quantile: True`)
   - Try other ranking methods (`probabilistic`, `smooth_i`)

