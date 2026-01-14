# Theoretical Bounds for Loss Components

## Overview

This document establishes theoretical bounds and expected ranges for each component of `ResearchAlignedICFLoss`. These bounds help us understand:
- Whether we're optimizing effectively
- If loss components are in reasonable ranges
- What "good" performance looks like for each component

## Notation

- **L_huber**: Huber loss component
- **L_rank**: Pairwise ranking loss component
- **L_spearman**: Spearman correlation loss (1 - ρ)
- **L_focal**: Focal loss component
- **L_asym**: Asymmetric penalty component
- **L_mono**: Monotonicity loss component
- **L_quantile**: Quantile regression loss component
- **ρ**: Spearman rank correlation coefficient
- **H(ICF)**: Shannon entropy of ICF distribution
- **I(Characters; ICF)**: Mutual information between character patterns and ICF

## 1. Spearman Loss

### Theoretical Bound

**Bound**: `1.0 - ρ_max`, where `ρ_max ≈ 0.18-0.19` for character-level models

**Expected Range**:
- **Best case**: `1.0 - 0.19 = 0.81`
- **Current best**: `1.0 - 0.1891 = 0.8109`
- **Typical**: `0.82 - 0.85`

**Interpretation**:
- Lower is better (closer to 0.81 = better)
- Values > 0.85 suggest poor ranking
- Values < 0.81 are theoretically impossible for character-level models

**Information-Theoretic Foundation**:
```
ρ_max ≤ sqrt(I(Characters; ICF) / H(ICF)) ≈ 0.18-0.19
```

## 2. Huber Loss

### Theoretical Bound

**Bound**: Depends on data distribution and model capacity

**Expected Range for ICF**:
- **Best case**: `0.01 - 0.05` (very small errors)
- **Good**: `0.05 - 0.10` (small errors)
- **Acceptable**: `0.10 - 0.20` (moderate errors)
- **Poor**: `> 0.20` (large errors)

**Mathematical Foundation**:
- Huber loss with δ=0.1: `L_δ(a) = 0.5a²` for |a| ≤ 0.1, else `0.1(|a| - 0.05)`
- For ICF values in [0, 1], typical errors are 0.1-0.3
- Expected loss: `E[L_δ] ≈ 0.05 - 0.15` for well-trained models

**Interpretation**:
- Lower is better
- Values < 0.05 indicate excellent regression accuracy
- Values > 0.20 suggest model is not learning well

**Connection to Spearman**:
- Low Huber loss doesn't guarantee high Spearman (ranking vs regression)
- But high Huber loss (> 0.20) usually indicates poor Spearman

## 3. Ranking Loss (Pairwise)

### Theoretical Bound

**Bound**: Depends on ranking method and data distribution

**Expected Range for ICF**:
- **Best case**: `0.0 - 0.05` (perfect pairwise ordering)
- **Good**: `0.05 - 0.15` (mostly correct ordering)
- **Acceptable**: `0.15 - 0.30` (some ordering errors)
- **Poor**: `> 0.30` (many ordering violations)

**Mathematical Foundation**:
- Margin-based ranking loss: `max(0, margin - pred_diff * sign(target_diff))`
- With margin=0.1, perfect ordering gives loss=0.0
- Random ordering gives loss ≈ 0.1 (margin value)
- For ICF with many pairs, expected loss scales with violation rate

**Theoretical Limitation**:
- Research shows: "All convex per-edge surrogates are inconsistent for strict ranking"
- This means even perfect optimization may not achieve loss=0.0
- Expected minimum: `0.02 - 0.05` due to theoretical inconsistency

**Interpretation**:
- Lower is better
- Values < 0.05 indicate excellent pairwise ranking
- Values > 0.30 suggest significant ranking errors
- Values between 0.05-0.15 are typical for well-trained models

**Connection to Spearman**:
- Ranking loss directly relates to Spearman correlation
- Low ranking loss (< 0.10) usually correlates with high Spearman (> 0.15)
- But ranking loss can be low even with moderate Spearman (local vs global ranking)

## 4. Focal Loss Component

### Theoretical Bound

**Bound**: Depends on base loss and gamma parameter

**Expected Range for ICF**:
- **Best case**: `0.01 - 0.05` (focused on hard examples, most are easy)
- **Good**: `0.05 - 0.15` (balanced focus)
- **Acceptable**: `0.15 - 0.30` (many hard examples)
- **Poor**: `> 0.30` (too many hard examples, model struggling)

**Mathematical Foundation**:
- Focal loss: `FL = (1 + error)^γ * base_loss`
- With γ=2.0, easy examples (error < 0.1) get down-weighted
- Hard examples (error > 0.3) get up-weighted exponentially
- For ICF, most words are "easy" (common words), few are "hard" (rare words)

**Expected Behavior**:
- Focal loss should be **lower** than base loss (Huber)
- If focal loss > Huber loss, focal weighting is not helping
- Typical ratio: `focal_loss / huber_loss ≈ 0.5 - 0.8` (focal down-weights easy examples)

**Interpretation**:
- Lower is better (but should be compared to base loss)
- Focal loss < Huber loss indicates focal weighting is effective
- Focal loss > Huber loss suggests γ is too high or model is struggling

## 5. Asymmetric Penalty

### Theoretical Bound

**Bound**: Depends on asymmetry factor and error distribution

**Expected Range for ICF**:
- **Best case**: `0.0 - 0.02` (minimal asymmetric errors)
- **Good**: `0.02 - 0.05` (some asymmetric errors, but small)
- **Acceptable**: `0.05 - 0.10` (moderate asymmetric errors)
- **Poor**: `> 0.10` (many large asymmetric errors)

**Mathematical Foundation**:
- Asymmetric penalty: `factor * relu(error)` for common→rare, `relu(-error)` for rare→common
- With factor=2.0, common→rare errors are penalized 2× more
- For ICF, common words (ICF ≈ 0.0) are more frequent, so common→rare errors are more common

**Expected Behavior**:
- Asymmetric penalty should be **smaller** than Huber loss (it's an additional penalty)
- Typical ratio: `asymmetric_penalty / huber_loss ≈ 0.1 - 0.3`
- If asymmetric penalty > Huber loss, asymmetry factor may be too high

**Interpretation**:
- Lower is better
- Values < 0.05 indicate model handles asymmetry well
- Values > 0.10 suggest model has systematic bias (over-predicting common words as rare)

## 6. Monotonicity Loss

### Theoretical Bound

**Bound**: Depends on feature-prediction correlation

**Expected Range for ICF**:
- **Best case**: `0.0` (perfect monotonicity)
- **Good**: `0.0 - 0.01` (minimal violations)
- **Acceptable**: `0.01 - 0.05` (some violations)
- **Poor**: `> 0.05` (many violations)

**Mathematical Foundation**:
- Monotonicity loss: `relu(-correlation)` for increasing, `relu(correlation)` for decreasing
- For word length → ICF (increasing), correlation should be positive
- For rare chars → ICF (decreasing), correlation should be negative
- Perfect monotonicity: correlation = ±1.0, loss = 0.0

**Expected Behavior**:
- Monotonicity loss should be **very small** (< 0.01) if constraints are reasonable
- If loss > 0.05, constraints may be too strict or model architecture is incompatible
- Typical values: `0.0 - 0.02` for well-trained models with reasonable constraints

**Interpretation**:
- Lower is better (0.0 is perfect)
- Values < 0.01 indicate good monotonicity
- Values > 0.05 suggest constraints are violated or too strict

## 7. Quantile Regression Loss

### Theoretical Bound

**Bound**: Depends on quantile and data distribution

**Expected Range for ICF**:
- **Best case**: `0.05 - 0.10` (good quantile prediction)
- **Good**: `0.10 - 0.20` (reasonable quantile prediction)
- **Acceptable**: `0.20 - 0.30` (moderate quantile errors)
- **Poor**: `> 0.30` (poor quantile prediction)

**Mathematical Foundation**:
- Quantile loss: `max(τ·error, (τ-1)·error)` for quantile τ
- For τ=0.5 (median), loss ≈ 0.5 * MAE
- For τ=0.9 (90th percentile), loss is asymmetric (over-prediction penalized more)
- Expected loss scales with prediction uncertainty

**Expected Behavior**:
- Quantile loss should be **comparable** to Huber loss (both measure regression error)
- Typical ratio: `quantile_loss / huber_loss ≈ 0.8 - 1.2`
- If quantile loss >> Huber loss, quantile prediction is poor

**Interpretation**:
- Lower is better
- Values < 0.10 indicate good quantile prediction
- Values > 0.30 suggest poor uncertainty estimation

## 8. Adaptive Regularization Strength

### Theoretical Bound

**Bound**: `1.0 / typical_difference`, where typical_difference is data scale

**Expected Range for ICF**:
- **Typical**: `5.0 - 20.0` (for ICF values in [0, 1])
- **High**: `20.0 - 50.0` (for very small differences)
- **Low**: `1.0 - 5.0` (for large differences)

**Mathematical Foundation**:
- Adaptive reg: `reg_strength = 1.0 / (std + MAD)`
- For ICF with std ≈ 0.2-0.3, MAD ≈ 0.1-0.2
- Expected: `reg_strength ≈ 1.0 / 0.2 = 5.0`
- Clamped to [0.1, 100.0] for stability

**Expected Behavior**:
- Should adapt to data scale automatically
- Values < 1.0 suggest data has very large differences (unlikely for ICF)
- Values > 50.0 suggest data has very small differences (possible for ICF)

**Interpretation**:
- Not "better" or "worse", just adaptive
- Values in [5.0, 20.0] are typical for ICF
- Values outside [1.0, 50.0] may indicate data issues

## Summary Table

| Component | Best Case | Good | Acceptable | Poor | Interpretation |
|-----------|-----------|------|------------|------|-----------------|
| **Spearman Loss** | 0.81 | 0.82-0.85 | 0.85-0.90 | > 0.90 | Lower is better, bound: 0.81 |
| **Huber Loss** | 0.01-0.05 | 0.05-0.10 | 0.10-0.20 | > 0.20 | Lower is better |
| **Ranking Loss** | 0.0-0.05 | 0.05-0.15 | 0.15-0.30 | > 0.30 | Lower is better, min: ~0.02 |
| **Focal Loss** | 0.01-0.05 | 0.05-0.15 | 0.15-0.30 | > 0.30 | Lower is better, < Huber |
| **Asymmetric Penalty** | 0.0-0.02 | 0.02-0.05 | 0.05-0.10 | > 0.10 | Lower is better, < Huber |
| **Monotonicity Loss** | 0.0 | 0.0-0.01 | 0.01-0.05 | > 0.05 | Lower is better, 0.0 = perfect |
| **Quantile Loss** | 0.05-0.10 | 0.10-0.20 | 0.20-0.30 | > 0.30 | Lower is better, ≈ Huber |
| **Reg Strength** | 5.0-20.0 | 1.0-50.0 | - | - | Adaptive, not "better/worse" |

## Practical Guidelines

### 1. Component Ratios

Monitor these ratios to understand loss balance:

- **Spearman / Total**: Should be dominant (0.7-0.9) since it's weighted 10×
- **Ranking / Total**: Should be moderate (0.05-0.15) since it's weighted 0.5×
- **Huber / Total**: Should be small (0.05-0.15) since it's base loss
- **Focal / Huber**: Should be < 1.0 (focal down-weights easy examples)
- **Asymmetric / Huber**: Should be < 0.5 (asymmetric is additional penalty)

### 2. Convergence Indicators

- **Spearman loss**: Should decrease to ~0.81-0.85 and stabilize
- **Huber loss**: Should decrease to ~0.05-0.10 and stabilize
- **Ranking loss**: Should decrease to ~0.05-0.15 and stabilize
- **Component ratios**: Should stabilize (not drift) during training

### 3. Warning Signs

- **Spearman loss > 0.90**: Model is not learning ranking
- **Huber loss > 0.20**: Model is not learning regression
- **Ranking loss > 0.30**: Many pairwise ordering violations
- **Focal loss > Huber loss**: Focal weighting not helping
- **Asymmetric penalty > 0.10**: Systematic bias in predictions
- **Monotonicity loss > 0.05**: Constraints violated or too strict

## Integration with Validation

These bounds are:
1. **Logged during validation** for each component (in `flexible_lightning_module.py`)
2. **Compared to bounds** automatically to identify issues
3. **Tracked over time** to monitor convergence
4. **Available for early stopping** if components don't improve

### Analysis Tools

Use `scripts/analyze_loss_bounds.py` to:
- Analyze bounds for all experiments
- Find optimization issues
- Compare component performance across experiments

```bash
# Analyze all experiments
python3 scripts/analyze_loss_bounds.py

# Analyze specific experiments
python3 scripts/analyze_loss_bounds.py --experiments iter7_roberta_best_loss iter7_bert_best_loss

# Find optimization issues
python3 scripts/analyze_loss_bounds.py --issues

# Verbose output
python3 scripts/analyze_loss_bounds.py --verbose
```

### Early Stopping Integration

Consider adding bounds-based early stopping:
- Stop if multiple components are "poor" for N epochs
- Stop if no component improves beyond "acceptable" for N epochs
- Stop if component ratios indicate optimization failure

See `src/tiny_icf/flexible_lightning_module.py` for current implementation.

