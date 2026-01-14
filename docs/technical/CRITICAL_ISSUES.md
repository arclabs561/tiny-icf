# Critical Issues Discovered

## Issue #1: Model Collapse (CRITICAL)

**Discovery**: Existing model `model_local_final.pt` has completely collapsed.

**Symptoms**:
- All predictions = 0.0 (100% collapse)
- Spearman = NaN (no variance)
- MAE = 0.55 (predicting 0.0 when targets are 0.15-0.62)
- Every word predicted as "Very Common" (0.0)

**Impact**: Model is completely unusable.

**Possible Causes**:
1. Output layer saturation/clamping to 0.0
2. Loss function encouraging collapse to mean
3. Training instability causing collapse
4. Model initialization issues

**Evidence**:
```json
{
  "pred_min": 0.0,
  "pred_max": 0.0,
  "pred_mean": 0.0,
  "pred_std": 0.0,
  "spearman_corr": NaN
}
```

**Fix Required**:
- Add collapse detection (monitor prediction variance)
- Review output layer (clamping, activation)
- Check loss function (may encourage collapse)
- Add early stopping on collapse

## Issue #2: Weak Ranking Signal

**Discovery**: New training run shows Spearman ~0.18 (barely better than random).

**Symptoms**:
- Spearman: 0.16 → 0.18 over 17 epochs (very slow)
- Would need 150+ epochs to reach 0.5
- Ranking loss may not be contributing effectively

**Impact**: Model cannot reliably rank words by commonality.

**Possible Causes**:
1. Ranking loss gradients too weak
2. Loss scale mismatch (Huber dominates)
3. Pair selection not providing strong signal
4. Model capacity insufficient

**Evidence**:
- Spearman improved only 0.014 over 6 epochs
- Model maintains full range [0,1] but ranking is wrong
- Training loss decreasing but ranking not improving proportionally

**Fix Required**:
- Log loss components separately
- Increase ranking weight (try 5.0, 10.0)
- Implement listwise ranking loss
- Add RBO (Rank-Biased Overlap) metric

## Issue #3: Training Instability

**Discovery**: Training speed varied dramatically (1-30 it/s).

**Symptoms**:
- Inconsistent iteration speed
- Possible memory pressure/swapping
- CPU throttling
- Background processes interfering

**Impact**: Makes training unreliable, hard to estimate completion.

**Fix Required**:
- Optimize data loading
- Check memory usage
- Reduce batch size if needed
- Monitor system resources

## Immediate Actions

1. **Add Collapse Detection**
   ```python
   # In training loop:
   if pred_std < 0.01:
       raise RuntimeError("Model collapsed - all predictions similar")
   ```

2. **Log Loss Components**
   ```python
   # Log separately:
   - huber_loss.item()
   - ranking_loss.item()
   - total_loss.item()
   ```

3. **Monitor Prediction Variance**
   ```python
   # Track over epochs:
   - pred_std (should be > 0.05)
   - pred_range (should be [0.0, 1.0])
   ```

4. **Run Ablation Study**
   - rank_weight=0 (Huber only)
   - rank_weight=2.0 (current)
   - rank_weight=10.0 (ranking heavy)

## Research Insights

From analysis of Spearman correlation issues:

1. **Non-Differentiability**: Spearman requires sorting (non-differentiable)
   - Need proxy losses (LambdaRank, ApproxNDCG)
   - Pairwise ranking may not be effective

2. **Top-K Bias**: Spearman can mask poor performance on top items
   - Need position-biased metrics (RBO)
   - Focus on ranking quality, not just correlation

3. **Loss Design**: Current pairwise ranking may not match Spearman
   - Need listwise losses
   - Need to emphasize top-ranked items

## Priority Fixes

**P0 (Critical)**:
1. Add collapse detection to prevent 0.0 predictions
2. Fix output layer to prevent saturation
3. Add prediction variance monitoring

**P1 (High)**:
1. Log loss components separately
2. Run ablation study on ranking weight
3. Implement listwise ranking loss

**P2 (Medium)**:
1. Add RBO evaluation metric
2. Test architecture variants
3. Validate data quality

