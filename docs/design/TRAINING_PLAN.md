# Training Plan: Fixing Collapse and Improving Ranking

## Overview

This plan implements the recommendations from `GOALS_CRITIQUE_AND_REFINEMENT.md` to:
1. Fix model collapse (all predictions = 0.0)
2. Improve ranking (Spearman currently 0.16-0.18)
3. Achieve Phase 1-2 goals (non-zero predictions, Spearman > 0.4)

## Immediate Actions

### 1. Diagnostic Training (Local or RunPod)

**Purpose**: Identify and fix collapse issues with detailed logging

**Script**: `scripts/train_diagnostic.py`

**Key Features**:
- Collapse detection (monitors prediction variance)
- Separate loss component logging (Huber vs Ranking)
- Improved initialization
- Higher ranking weight (5.0, 10.0 options)
- Early stopping on collapse

**Run Locally**:
```bash
# Single experiment
uv run scripts/train_diagnostic.py \
    --data data/word_frequency.csv \
    --epochs 30 \
    --batch-size 64 \
    --rank-weight 5.0 \
    --huber-delta 0.2 \
    --output models/model_diagnostic.pt \
    --history training_history/diagnostic.json

# Or run all experiments
./scripts/train_local_experiments.sh
```

**Run on RunPod**:
```bash
# Upload and start training
./scripts/train_runpod_diagnostic.sh <pod-id>

# Monitor
ssh -i ~/.ssh/id_ed25519 -p 31179 root@38.80.152.76 \
    'tail -f /root/idf-est/training_history/runpod_diagnostic.log'
```

### 2. Experiments to Run

**Experiment 1: Diagnostic (rank_weight=5.0)**
- Baseline with increased ranking weight
- Should show if ranking loss is contributing
- Monitor collapse diagnostics

**Experiment 2: Higher Ranking (rank_weight=10.0)**
- Even stronger ranking signal
- Compare with Experiment 1
- May show if ranking loss helps

**Experiment 3: Best Practices Baseline**
- Compare with current best practices script
- Establish baseline for comparison

## Success Criteria

### Phase 1: Fix Collapse (Immediate)
- ✅ Model produces non-zero predictions
- ✅ Predictions span meaningful range (not all 0.0 or all 1.0)
- ✅ Spearman > 0.1 (better than random)

### Phase 2: Basic Learning (Short-term)
- MAE < 0.25 (high-freq words), < 0.40 (full vocab)
- Spearman correlation > 0.4 (shows ranking ability)
- Jabberwocky Protocol: 3/5+ tests pass

## What to Monitor

### During Training

1. **Collapse Detection**:
   - `pred_std` should be > 0.01
   - `pred_range` should be > 0.1
   - `zero_fraction` should be < 0.95

2. **Loss Components**:
   - `huber_loss`: Should decrease over time
   - `ranking_loss`: Should decrease (indicates ranking learning)
   - Ratio: ranking_loss / huber_loss (should be meaningful)

3. **Metrics**:
   - Spearman: Should improve (target: > 0.4)
   - MAE: Should decrease (target: < 0.25)
   - Jabberwocky: Should pass 3/5+ tests

### Red Flags

- **Collapse detected**: Stop training, investigate initialization
- **Ranking loss = 0**: Pairs not being generated correctly
- **Ranking loss not decreasing**: Loss weight too low or pairs not informative
- **Spearman not improving**: May need architecture changes or more training

## Next Steps After Experiments

### If Collapse Fixed:
1. Continue training with best hyperparameters
2. Try different architectures (HierarchicalICF, BoxEmbeddingICF)
3. Experiment with listwise losses

### If Collapse Persists:
1. Review initialization (may need different strategy)
2. Check output layer (clamp may be too aggressive)
3. Try sigmoid instead of clamp
4. Review loss function (may encourage collapse)

### If Ranking Still Weak:
1. Try even higher ranking weights (20.0, 50.0)
2. Implement listwise ranking loss
3. Try differentiable sorting (diffsort)
4. Consider architecture changes

## Files Created

- `scripts/train_diagnostic.py`: Diagnostic training with collapse detection
- `scripts/train_local_experiments.sh`: Local experiment runner
- `scripts/train_runpod_diagnostic.sh`: RunPod training script
- `TRAINING_PLAN.md`: This file

## References

- `GOALS_CRITIQUE_AND_REFINEMENT.md`: Evidence-based goals and recommendations
- `CRITICAL_ISSUES.md`: Known issues (collapse, weak ranking)
- `EXPERIENCE_AND_CRITIQUE.md`: Previous training observations

