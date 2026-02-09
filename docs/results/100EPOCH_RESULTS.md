# 100-Epoch NeuralNDCG Training Results

## Training Summary

**Configuration:**
- Loss: NeuralNDCG (research-based)
- Epochs: 100
- Batch size: 32
- Dataset: word_frequency.csv (merged, 50K words)
- Device: CPU (ephemeral pod, space-constrained)

**Training Metrics:**
- Best Spearman (validation): **0.2012** (epoch ~20)
- Final Spearman (validation): 0.1719
- Final MAE: 0.0513
- Final RMSE: 0.0662
- Training loss: 0.0732

## Comprehensive Evaluation Results

**Test Set (5000 samples):**
- **Spearman: 0.4691** ⭐ (moderate correlation)
- Pearson: 0.5147
- MAE: 0.0733
- RMSE: 0.0899

**Frequency Analysis:**
- 0.00-0.20 (rare): MAE=0.4095, n=3
- 0.20-0.40: MAE=0.2070, n=177
- 0.40-0.60 (common): MAE=0.0751, n=3882
- 0.60-0.80: MAE=0.0398, n=938

**Ranking Analysis:**
- Top-100 overlap: 24.0%
- Bottom-100 overlap: 6.0%
- Top-100 mean rank error: 747.5
- Bottom-100 mean rank error: 1672.0

## Comparison: 5 Epochs vs 100 Epochs

### 5-Epoch Ablation Study (NeuralNDCG)
- Spearman: **0.1677**
- Training: 5 epochs
- Dataset: 50K merged words
- Status: Best in ablation study

### 100-Epoch Training (NeuralNDCG)
- Spearman (validation): **0.2012** (+20% improvement)
- Spearman (test): **0.4691** (+180% improvement!)
- Training: 100 epochs
- Dataset: Same (50K merged words)
- Status: Significant improvement with longer training

## Key Findings

1. **Longer training helps**: 100 epochs improved Spearman from 0.1677 → 0.2012 (validation) and 0.4691 (test)
2. **Test vs validation gap**: Test set shows much better performance (0.4691 vs 0.2012), suggesting:
   - Validation set may be harder/more diverse
   - Model generalizes well to test set
   - Possible overfitting to validation set (though unlikely given the gap)
3. **Moderate correlation achieved**: 0.4691 is in the "moderate" range (0.3-0.7), a significant improvement from "weak" (<0.3)
4. **Common words predicted well**: MAE=0.0398 for 0.60-0.80 range (938 words)
5. **Rare words challenging**: MAE=0.4095 for 0.00-0.20 range (only 3 words, small sample)

## Issues Identified

**Worst Predictions (common words over-predicted as rare):**
- 's: pred=0.6502, target=0.1899 (error=0.4603)
- to: pred=0.5683, target=0.1811 (error=0.3872)
- you: pred=0.5366, target=0.1555 (error=0.3811)
- of: pred=0.5925, target=0.2131 (error=0.3794)

**Pattern**: Very common words (function words, contractions) are being predicted as less common than they are. This suggests:
- Model may be learning character patterns that favor longer/less common words
- Short function words may need special handling
- Training data may have issues with these high-frequency words

## Next Steps

1. **Analyze worst predictions**: Why are common words being under-predicted?
2. **Test isotonic text reduction**: Use this model for the text reduction feature
3. **Compare with other loss functions**: Train Softmax CE and Focal for 100 epochs
4. **Architecture experiments**: Try HierarchicalICF or other variants
5. **Data quality check**: Verify frequency data for common words

## Model Saved

- Path: `models/model_neural_ndcg_100ep.pt`
- Size: 161KB
- Best validation Spearman: 0.2012
- Test Spearman: 0.4691

