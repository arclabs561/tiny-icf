# Research Quick Reference

## Top 5 Actionable Improvements

### 1. Dynamic Temperature Scheduling ⭐⭐⭐
**Impact**: High | **Effort**: Low | **Priority**: P0

Replace static temperature with divergence-based adjustment:
```python
temp = base_temp * (1 + divergence / teacher_loss)
```

### 2. Soft Ranking Loss ⭐⭐⭐
**Impact**: High | **Effort**: Low | **Priority**: P0

We already have `rank-relax.spearman_loss_pytorch` - just add it:
```python
spearman_loss = rank_relax.spearman_loss_pytorch(pred, target, regularization_strength=1e-2)
total_loss = 0.7 * mse_loss + 0.2 * distillation_loss + 0.1 * spearman_loss
```

### 3. Hierarchical Feature Alignment ⭐⭐
**Impact**: Medium | **Effort**: Medium | **Priority**: P1

Multiple teacher layers → single student layer with learned weights:
```python
weighted_features = sum(w * align(t) for w, t in zip(attention_weights, teacher_layers))
```

### 4. Attention Mechanism ⭐⭐
**Impact**: Medium | **Effort**: Medium | **Priority**: P1

Add multi-head self-attention after CNN layers:
```python
self.attention = nn.MultiheadAttention(embed_dim=features, num_heads=4)
```

### 5. RankDistil Listwise Loss ⭐
**Impact**: Medium | **Effort**: High | **Priority**: P2

Preserve teacher's ranking order:
```python
# Penalize violations of teacher's ranking
for i, j in teacher_ranked_pairs:
    if teacher[i] > teacher[j] and student[i] <= student[j]:
        loss += margin - (student[j] - student[i])
```

---

## Key Research Findings

### Temperature Scaling
- **Static**: 3-5 (baseline)
- **Dynamic**: Adjust based on student-teacher divergence (2-10 range)
- **Asymmetric**: Different temps for correct/incorrect classes

### Feature Alignment
- **Fixed**: Student layer N → Teacher layer M (current)
- **Hierarchical**: Multiple teacher layers → single student (better)
- **Adaptation layers**: Critical for token→character bridging

### Loss Weighting
- **Hard loss (α)**: 0.5-0.7 initially, decreases over time
- **Soft loss (β)**: 0.3-0.5 initially, increases over time
- **Feature loss (γ)**: 0.1-0.2 (auxiliary)

### Spearman Optimization
- **Problem**: Sort operation not differentiable
- **Solution**: Soft ranking approximation (`rank-relax`)
- **Trade-off**: Regularization strength vs. gradient flow

### Architecture
- **Residual connections**: Already have `ResidualICF` ✅
- **Attention**: Add after CNN layers for long-range dependencies
- **Listwise > Pairwise**: Better preserves ranking structure

---

## Performance Expectations

**Current**: Spearman ~0.17

**With Dynamic Temp + Feature Alignment**: 0.20-0.22 (+0.03-0.05)

**With Soft Ranking Loss**: 0.22-0.25 (+0.02-0.03)

**With Attention**: 0.24-0.28 (+0.02-0.03)

**Target**: 0.25-0.30 Spearman

**Research Benchmark**: 95.8-97.7% of teacher performance retained

---

## Implementation Checklist

### Quick Wins (1-2 days)
- [ ] Dynamic temperature scheduling
- [ ] Soft ranking loss (using rank-relax)
- [ ] Enhanced logging

### Architecture (3-5 days)
- [ ] Attention mechanism
- [ ] Validation testing
- [ ] Hyperparameter tuning

### Advanced (1 week)
- [ ] Hierarchical feature alignment
- [ ] RankDistil listwise loss
- [ ] Margin-aware contrastive learning

---

## Key Papers

1. **"Bridging the Gap: Knowledge Distillation for Online Ranking Systems"** (2024)
   - Directly addresses our use case
   - Identifies unique ranking challenges

2. **"An Empirical Study of Uniform-Architecture Knowledge Distillation"** (2023)
   - 7-24× speedup, 95.8-97.7% performance retained
   - Validates our approach

3. **"PLD: Choice-Theoretic List-Wise Knowledge Distillation"** (2025)
   - Listwise > pairwise for ranking

---

**Next Steps**: Start with Priority P0 items (Dynamic Temp + Soft Ranking Loss)

