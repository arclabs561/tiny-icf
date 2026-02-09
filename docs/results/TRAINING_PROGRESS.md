# Training Progress: Diagnostic Experiments

## Current Status

**Training Started**: Diagnostic experiment with `rank_weight=5.0`

## Initial Results (5 epochs)

### Metrics Progression
- **Epoch 1**: MAE=0.2927, Spearman=0.1051, Pred_std=0.3490
- **Epoch 2**: MAE=0.2741, Spearman=0.1499, Pred_std=0.3685
- **Epoch 4**: MAE=0.2938, Spearman=0.1636, Pred_std=0.3697 (best)

### Key Observations

**✅ No Collapse Detected**
- Prediction std: 0.35-0.37 (healthy variance)
- Predictions span full range [0.0, 1.0]
- No collapse events logged

**✅ Ranking Loss Working**
- Ranking loss decreasing: 14.97 → 11.61 (over 5 epochs)
- Ranking loss is contributing (not zero)
- Ratio: ranking_loss / huber_loss ≈ 240:1 (ranking dominates as intended)

**✅ Spearman Improving**
- Spearman: 0.105 → 0.164 in 5 epochs
- Improvement rate: ~0.012 per epoch
- Much faster than previous training (0.002 per epoch)

**⚠️ MAE Not Improving**
- MAE: 0.29-0.27 (fluctuating, not decreasing)
- Still above Phase 2 target (< 0.25)
- May need more training or different approach

## Comparison to Previous Training

| Metric | Previous (17 epochs) | Current (5 epochs) | Improvement |
|--------|----------------------|-------------------|-------------|
| Spearman | 0.16-0.18 | 0.16 | Similar, but faster |
| MAE | 0.29-0.32 | 0.27-0.29 | Slightly better |
| Collapse | None detected | None detected | ✅ |
| Ranking loss | Unknown | 11.6 (decreasing) | ✅ Tracked |

## Next Steps

1. **Continue Current Training**: Let it run to 30 epochs, monitor for:
   - Spearman > 0.4 (Phase 2 target)
   - MAE < 0.25 (Phase 2 target)
   - No collapse events

2. **Try Higher Ranking Weight**: If Spearman plateaus, try `rank_weight=10.0`

3. **Compare Experiments**: Run all three experiments and compare results

## Goals Alignment

### Phase 1: Fix Collapse ✅
- ✅ Model produces non-zero predictions
- ✅ Predictions span meaningful range
- ✅ Spearman > 0.1 (currently 0.16)

### Phase 2: Basic Learning (In Progress)
- ⚠️ MAE < 0.25 (currently 0.27-0.29, close!)
- ⚠️ Spearman > 0.4 (currently 0.16, improving)
- ⏳ Jabberwocky: Not tested yet

## Files

- **Model**: `models/test_diagnostic.pt` (5 epochs)
- **Full Training**: `models/model_diagnostic_rank5.pt` (30 epochs, running)
- **History**: `training_history/diagnostic_rank5.json`
- **Log**: `training_history/diagnostic_rank5.log`

