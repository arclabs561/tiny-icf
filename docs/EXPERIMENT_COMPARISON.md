# Experiment Comparison: rank_weight=5.0 vs 10.0

## Training Metrics

### rank_weight=5.0
- **Best Spearman**: 0.1733 (epoch 24)
- **Best MAE**: 0.2725
- **Ranking Loss**: 15.69 → 9.58 (38.9% reduction)
- **Collapse Events**: 0
- **Prediction Std**: 0.32-0.38 (healthy)

### rank_weight=10.0
- **Best Spearman**: 0.1736 (epoch 24)
- **Best MAE**: 0.2739
- **Ranking Loss**: 15.63 → 9.59 (38.6% reduction)
- **Collapse Events**: 0
- **Prediction Std**: 0.32-0.37 (healthy)

**Training Conclusion**: Very similar performance in training metrics.

## Full Dataset Evaluation

### rank_weight=5.0
- **Spearman**: 0.2842 (p < 0.001)
- **MAE**: 0.3103
- **Jabberwocky**: 40.0% (2/5)
- **Prediction Mean**: 0.1325 (target: 0.4042)
- **Prediction Std**: 0.2061

### rank_weight=10.0
- **Spearman**: 0.2354 (p < 0.001)
- **MAE**: 0.3296
- **Jabberwocky**: 80.0% (4/5) ✅
- **Prediction Mean**: 0.0965 (target: 0.4042)
- **Prediction Std**: 0.1721

## Key Findings

### rank_weight=5.0 Advantages
1. ✅ **Better Spearman** on full dataset (0.2842 vs 0.2354, +21%)
2. ✅ **Better MAE** (0.3103 vs 0.3296, -6%)
3. ✅ **Less under-prediction** (mean=0.13 vs 0.10)

### rank_weight=10.0 Advantages
1. ✅ **Excellent Jabberwocky** (80% vs 40%, +100%)
2. ✅ **Better rare word detection** (correctly identifies rare but valid words)
3. ✅ **Better at distinguishing** common vs rare words

## Analysis

### Why rank_weight=10.0 Has Lower Spearman?

**Hypothesis**: Higher ranking weight emphasizes relative ordering over absolute values, which:
- ✅ Helps with rare word detection (Jabberwocky)
- ⚠️ May hurt overall correlation (Spearman)
- ⚠️ Increases under-prediction (mean=0.10 vs 0.13)

The model is learning to distinguish rare vs common better, but at the cost of absolute accuracy.

### Why rank_weight=5.0 Has Better Spearman?

**Hypothesis**: Lower ranking weight balances:
- ✅ Absolute accuracy (Huber loss)
- ✅ Relative ordering (Ranking loss)
- ✅ Better overall correlation

### Under-Prediction Issue

Both models severely under-predict:
- rank_weight=5.0: mean=0.13 (67% under)
- rank_weight=10.0: mean=0.10 (75% under)

**This is the main issue to address** - calibration loss should help.

## Recommendations

### For Different Use Cases

1. **Rare Word Detection** (Jabberwocky-like tasks):
   - Use **rank_weight=10.0**
   - Better at distinguishing rare vs common
   - 80% Jabberwocky pass rate

2. **Overall Correlation** (General ICF prediction):
   - Use **rank_weight=5.0**
   - Better Spearman (0.28 vs 0.24)
   - Better MAE (0.31 vs 0.33)

3. **Best of Both Worlds**:
   - Use **calibrated training** (addresses under-prediction)
   - Should improve both Spearman and Jabberwocky
   - Currently running...

## Next Steps

1. ⏳ **Wait for calibrated training** to complete
2. **Evaluate calibrated model** on full dataset
3. **Compare all three** (rank_weight=5.0, 10.0, calibrated)
4. **Iterate** on best approach

## Expected Improvements from Calibration

Calibrated training should:
- ✅ Fix under-prediction (prediction mean → 0.40)
- ✅ Improve Spearman (better distribution matching)
- ✅ Improve Jabberwocky (better rare word detection)
- ✅ Maintain or improve MAE

**Target**: Spearman >0.30, Jabberwocky >60%, MAE <0.30

