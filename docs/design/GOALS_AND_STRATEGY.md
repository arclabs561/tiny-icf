# Goals, Strategy, and Multi-Loss Training

## Our Goals and Why

### Primary Goal
**Build a tiny (< 20k params), powerful model that predicts word frequency (ICF) for zero-shot text processing.**

### Why This Matters

#### 1. **Token Filtering in RAG/Retrieval**
- **Problem**: Embedding computation is expensive (30-50% of RAG cost)
- **Solution**: Filter stopwords before embedding
- **Impact**: 30-50% cost reduction
- **Requirement**: Fast inference (< 1ms/word), small model (< 80KB)

#### 2. **Zero-Shot Classification/Retrieval**
- **Problem**: Need to weight tokens by informativeness without training
- **Solution**: Use ICF to down-weight common words, up-weight rare content words
- **Impact**: Better retrieval quality without domain-specific training
- **Requirement**: Accurate relative ranking (common < rare)

#### 3. **Text Quality Assessment**
- **Problem**: Detect gibberish, low-quality content
- **Solution**: Very high ICF for random strings = gibberish
- **Impact**: Filter spam, detect encoding errors
- **Requirement**: High ICF for impossible structures (qzxbjk → 0.99)

### Success Metrics

1. **Accuracy**
   - MAE < 0.1 (mean absolute error)
   - Spearman correlation > 0.8 (ranking accuracy)
   - Jabberwocky Protocol: 4/5 tests pass

2. **Size**
   - Parameters < 20k
   - Model size < 80 KB (float32) / 40 KB (float16)
   - Binary size < 2 MB (with weights)

3. **Speed**
   - Inference < 1ms per word (CPU)
   - Throughput > 1000 words/sec

4. **Generalization**
   - Handles all languages (byte-level)
   - Handles typos/misspellings
   - Handles neologisms/portmanteaus
   - Handles out-of-vocabulary words

## Current Training Strategy

### Single Model Training
- **Architecture**: UniversalICF (~40k params) or variants
- **Loss**: Combined Huber + Ranking
- **Data**: word_frequency.csv (~50k words)
- **Status**: Training in progress (Epoch 16/100)

### Current Loss Function
```python
CombinedLoss = HuberLoss + RankingLoss
- HuberLoss: Smooth L1 for absolute ICF values
- RankingLoss: Ensures common < rare ordering
```

**Limitations:**
- Only 2 loss components
- Ranking loss may not be strong enough
- No explicit contrastive learning
- No calibration for frequency matching

## Multi-Loss Training Strategy

### Option 1: Enhanced Multi-Loss (Recommended)

```python
TotalLoss = (
    w1 * HuberLoss +           # Absolute values
    w2 * RankingLoss +         # Relative ordering
    w3 * ContrastiveLoss +     # Push common/rare apart
    w4 * ConsistencyLoss +     # Similar words → similar ICF
    w5 * CalibrationLoss       # Match actual frequencies
)
```

**Benefits:**
- Better ranking (contrastive)
- Better generalization (consistency)
- Better calibration (frequency matching)

**Trade-offs:**
- More hyperparameters to tune
- Slower training (more loss computation)
- Need to balance weights

### Option 2: Curriculum Multi-Loss

**Stage 1 (Early epochs)**: Focus on absolute values
```python
Loss = HuberLoss + 0.5 * RankingLoss
```

**Stage 2 (Middle epochs)**: Add contrastive learning
```python
Loss = HuberLoss + RankingLoss + 0.5 * ContrastiveLoss
```

**Stage 3 (Late epochs)**: Add calibration
```python
Loss = HuberLoss + RankingLoss + ContrastiveLoss + CalibrationLoss
```

**Benefits:**
- Progressive learning
- Easier to tune
- Better convergence

### Option 3: Task-Specific Multi-Loss

Different losses for different word types:
- **Common words**: Focus on absolute accuracy (Huber)
- **Rare words**: Focus on ranking (Ranking + Contrastive)
- **Out-of-vocab**: Focus on consistency (Consistency)

## Training Multiple Variations

### Why Train Multiple Variations?

1. **Architecture Search**: Find best model size/performance trade-off
2. **Ensemble**: Combine predictions for better accuracy
3. **Ablation Studies**: Understand what works
4. **Robustness**: Different models may handle different cases better

### Variations to Train

#### 1. **Size Variations**
- UniversalICF (40k params) - baseline
- HierarchicalICF (16k params) - smaller, hierarchical
- BoxEmbeddingICF (14k params) - smallest, box embeddings
- NanoICF (6.7k params) - ultra-tiny

#### 2. **Loss Variations**
- Baseline: Huber + Ranking
- Enhanced: + Contrastive + Consistency
- Curriculum: Progressive loss addition
- Task-specific: Different losses per word type

#### 3. **Data Variations**
- word_frequency.csv (original)
- word_frequency_modern.csv (+ modern words)
- combined_frequencies.csv (more data)
- multilingual_combined.csv (multilingual)

### Training Strategy

**Phase 1: Architecture Search** (Current)
1. Train UniversalICF (baseline)
2. Train HierarchicalICF
3. Train BoxEmbeddingICF
4. Compare size vs performance

**Phase 2: Loss Optimization**
1. Train with enhanced multi-loss
2. Train with curriculum multi-loss
3. Compare loss strategies

**Phase 3: Data Optimization**
1. Train with modern words
2. Train with more data
3. Compare data strategies

**Phase 4: Final Selection**
1. Choose best architecture
2. Choose best loss strategy
3. Choose best data
4. Train final model

## Recommended Next Steps

### Immediate (Continue Current Training)
1. ✅ Let current training finish (100 epochs)
2. ✅ Test on Jabberwocky Protocol
3. ✅ Measure metrics (MAE, Spearman, etc.)

### Short-term (Multi-Loss)
1. Implement enhanced multi-loss function
2. Add contrastive loss (push common/rare apart)
3. Add consistency loss (similar words → similar ICF)
4. Train with multi-loss

### Medium-term (Multiple Variations)
1. Train HierarchicalICF variant
2. Train BoxEmbeddingICF variant
3. Compare all variants
4. Choose best architecture

### Long-term (Optimization)
1. Quantization (float16, int8)
2. Pruning (remove low-importance weights)
3. Knowledge distillation (small from large)
4. Final deployment model

## Key Questions Answered

### Q: Why multi-loss training?
**A**: Single loss (Huber + Ranking) may not capture all aspects:
- Ranking loss ensures ordering but may not push common/rare far enough
- No explicit learning of word similarity
- No calibration to actual frequencies

### Q: Why train multiple variations?
**A**: 
- Find best size/performance trade-off
- Different architectures may excel at different tasks
- Ensemble can improve accuracy
- Ablation studies to understand what works

### Q: What are our goals?
**A**: 
1. **Tiny model** (< 20k params) for deployment
2. **Powerful** (accurate ICF prediction)
3. **Fast** (< 1ms inference)
4. **General** (handles all languages, typos, neologisms)

### Q: Why these goals?
**A**: 
- **Tiny**: Fits in L2 cache, fast inference, easy deployment
- **Powerful**: Accurate filtering/weighting improves downstream tasks
- **Fast**: Real-time processing in RAG/retrieval systems
- **General**: Zero-shot, no domain-specific training needed

