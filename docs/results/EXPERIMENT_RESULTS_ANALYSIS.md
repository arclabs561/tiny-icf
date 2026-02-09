# Experiment Results Analysis

## Completed Experiments

### 1. Reduced Capacity Model
- **Parameters**: 24,895 (37.9% reduction from original)
- **Config**: emb=36, conv=18, hidden=36, dropout=0.4
- **Training**: Early stopped at epoch 28 (no improvement for 15 epochs)
- **Best Validation Spearman**: 0.0661
- **Final Evaluation Metrics**:
  - Spearman: 0.0482
  - MAE: 0.3839
  - RMSE: 0.4250
  - Separation: 0.0532
- **Observations**: 
  - Lower MAE than BatchNorm (better absolute accuracy)
  - Lower Spearman than BatchNorm (worse ranking)
  - Model learned but validation performance limited

### 2. BatchNorm Model
- **Parameters**: 25,003 (BatchNorm added)
- **Config**: emb=36, conv=18, hidden=36, dropout=0.4, BatchNorm=True
- **Training**: Early stopped at epoch 41 (no improvement for 15 epochs)
- **Best Validation Spearman**: 0.0728
- **Final Evaluation Metrics**:
  - Spearman: 0.0633
  - MAE: 0.4362
  - RMSE: 0.4812
  - Separation: 0.0832
- **Observations**:
  - Higher Spearman than Reduced Capacity (better ranking)
  - Higher MAE than Reduced Capacity (worse absolute accuracy)
  - Better separation between common and rare words
  - Trained longer (41 epochs vs 28)

### 3. Temporal AMOO (Baseline)
- **Parameters**: ~40,127 (original architecture)
- **Best Validation Spearman**: 0.1335
- **Final Train Spearman**: 0.3885
- **Final Val Spearman**: 0.1333
- **Gap**: 0.2552 (65.7%)
- **Observations**:
  - Best overall performance
  - Still significant overfitting gap
  - Used temporal data and AMOO

## Key Findings

### 1. Capacity Reduction Impact
- **37.9% parameter reduction** did not significantly improve generalization
- Validation Spearman dropped from 0.1335 (baseline) to 0.0661 (reduced)
- Overfitting gap remains large (~70%+)

### 2. BatchNorm Impact
- **Slight improvement** in ranking (Spearman 0.0633 vs 0.0482)
- **Worse absolute accuracy** (MAE 0.4362 vs 0.3839)
- **Better separation** between common and rare words (0.0832 vs 0.0532)
- Trained longer before early stopping

### 3. Overfitting Persists
- All models show large train/val gaps (65-70%+)
- Regularization (dropout=0.4, weight_decay=1e-4) insufficient
- Models learn training patterns but fail to generalize

## Recommendations

### Immediate Next Steps
1. **Test Residual Model**: ResidualICF (30,943 params) ready to train
   - Hypothesis: Residual connections improve gradient flow
   - May help with learning and generalization

2. **More Aggressive Regularization**:
   - Increase dropout to 0.5-0.6
   - Increase weight decay to 1e-3
   - Add label smoothing
   - Increase augmentation probability to 0.3-0.4

3. **Architecture Experiments**:
   - Try even smaller models (NanoICF: 6,721 params)
   - Test gated residual connections
   - Experiment with different pooling strategies

4. **Training Strategy**:
   - Longer warmup periods
   - Different learning rate schedules
   - Mixup or other data augmentation techniques
   - Ensemble approaches

### Best Model So Far
- **Temporal AMOO** remains the best performer (Spearman 0.1335)
- Despite overfitting, it achieves best validation performance
- Consider using this as baseline for further improvements

## Metrics Summary

| Model | Spearman | MAE | RMSE | Separation | Best Val Spearman |
|-------|----------|-----|------|------------|-------------------|
| Temporal AMOO | 0.1335 | - | - | - | 0.1335 |
| BatchNorm | 0.0633 | 0.4362 | 0.4812 | 0.0832 | 0.0728 |
| Reduced Capacity | 0.0482 | 0.3839 | 0.4250 | 0.0532 | 0.0661 |

**Winner**: Temporal AMOO (baseline) > BatchNorm > Reduced Capacity
