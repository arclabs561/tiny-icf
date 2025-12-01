# Training Analysis and Results

## Initial Training Run

### Dataset
- **Words**: 103 unique words
- **Total tokens**: 8,835,249
- **Training samples**: 33 (after 80/20 split)
- **Validation samples**: 8

### Model Architecture
- **Parameters**: 53,473 (slightly over 50k target)
- **Architecture**: Byte-level CNN with parallel kernels (3, 5, 7)
- **Output**: Sigmoid-normalized ICF (0.0 = common, 1.0 = rare)

### Training Configuration
- **Epochs**: 20
- **Batch size**: 32
- **Learning rate**: 1e-3
- **Loss**: Combined Huber + Ranking
- **Augmentation**: 10% character-level noise

### Results

#### Performance Metrics
- **MAE**: 0.1486
- **RMSE**: 0.1806
- **Spearman Correlation**: -0.1822 (p=0.0655) ⚠️ **NEGATIVE CORRELATION**
- **Top-10 Ranking Accuracy**: 0/10 words correctly identified

#### Critical Issues Identified

1. **Near-Constant Predictions**
   - Prediction std: 0.0119 (very low variance)
   - Ground truth std: 0.1773 (expected variance)
   - Model is predicting ~0.44 for almost all words

2. **Negative Correlation**
   - Spearman correlation is negative, meaning model is learning the *opposite* of what it should
   - This suggests fundamental learning problem

3. **Poor Ranking**
   - Model cannot distinguish between common words (0.13) and rare words (1.0)
   - All predictions clustered around 0.4-0.45

### Root Cause Analysis

#### Primary Issue: Dataset Too Small
- **103 words** is insufficient for a character-level model to learn patterns
- After stratification and train/val split: only **33 training samples**
- Model has **53k parameters** but only **33 examples** → severe overfitting risk

#### Secondary Issues
1. **Limited diversity**: Most words are common (ICF 0.13-0.68), few rare examples
2. **No morphological patterns**: Small dataset doesn't expose enough patterns for CNN to learn
3. **Training instability**: With so few samples, model may converge to trivial solution

### Recommendations

#### Immediate Fixes

1. **Reduce Model Size** (to get under 50k)
   - Reduce embedding dim: 64 → 48
   - Reduce conv channels: 32 → 24
   - Reduce hidden dim: 64 → 48
   - Expected: ~30k parameters

2. **Increase Dataset Size**
   - Need at least 1,000-10,000 words for meaningful learning
   - Better: 100k+ words from real corpus

3. **Adjust Training**
   - More epochs (50-100)
   - Lower learning rate (1e-4)
   - Stronger regularization (dropout 0.2-0.3)
   - Weight decay for L2 regularization

#### Long-term Solutions

1. **Use Real Frequency Data**
   - Google 1T Word Corpus
   - Common Crawl derived lists
   - Wikipedia frequency lists

2. **Curriculum Learning**
   - Start with common words
   - Gradually introduce rare words
   - Helps model learn structure incrementally

3. **Better Loss Function**
   - Add explicit ranking loss with more pairs
   - Use contrastive loss for rare vs common
   - Weight rare examples more heavily

### Next Steps

1. ✅ Reduce model size to <50k parameters
2. Generate larger test dataset (1000+ words)
3. Retrain with improved hyperparameters
4. Validate Jabberwocky Protocol
5. Measure Spearman correlation (target: >0.8)

### Expected Improvements

With larger dataset (1000+ words):
- **MAE**: Should drop to <0.05
- **Spearman**: Should be >0.8 (positive correlation)
- **Ranking**: Top-10 accuracy should be >8/10
- **Jabberwocky**: Should pass all test cases

### Model Size Optimization

After reducing dimensions:
- Embedding: 256 × 48 = 12,288
- Conv3: 48 × 24 × 3 = 3,456
- Conv5: 48 × 24 × 5 = 5,760
- Conv7: 48 × 24 × 7 = 8,064
- Head: (72 × 48) + (48 × 1) = 3,456 + 48 = 3,504
- **Total**: ~33k parameters (well under 50k)

