# Unfinished Threads & Pending Work

Comprehensive list of features, experiments, and improvements that were started but not completed.

## 🔴 High Priority - Core Features

### 1. **Model Architecture Variants** (Partially Complete)
**Status**: Implemented but not trained/tested

- ✅ `HierarchicalICF` - Implemented in `src/tiny_icf/model_hierarchical.py`
- ✅ `BoxEmbeddingICF` - Implemented in `src/tiny_icf/model_hierarchical.py`
- ✅ `NanoICF` - Implemented in `src/tiny_icf/nano_model.py`
- ✅ `train_variations.py` - Script exists but needs testing
- ❌ **Not Done**: Train and compare all variants
- ❌ **Not Done**: Performance comparison (size vs accuracy)
- ❌ **Not Done**: Choose best architecture

**Next Steps**:
```bash
# Train all variants
python scripts/train_variations.py --data data/word_frequency.csv --epochs 50

# Compare results
python scripts/compare_training.py \
    --model1 models/variations/universal.pt \
    --model2 models/variations/hierarchical.pt
```

### 2. **Multi-Loss Training** (Partially Complete)
**Status**: Implemented but not fully tested

- ✅ `EnhancedMultiLoss` - Implemented in `src/tiny_icf/loss_multi.py`
  - Components: Huber + Ranking + Contrastive + Consistency + Calibration
- ✅ `CurriculumMultiLoss` - Progressive loss addition
- ✅ `train_multi_loss.py` - Training script exists
- ❌ **Not Done**: Validate multi-loss improves over baseline
- ❌ **Not Done**: Tune loss weights
- ❌ **Not Done**: Compare multi-loss vs standard training

**Next Steps**:
```bash
# Train with multi-loss
python -m tiny_icf.train_multi_loss \
    --data data/word_frequency.csv \
    --epochs 100 \
    --multi-loss \
    --output models/model_multi.pt

# Compare with baseline
python scripts/compare_training.py \
    --model1 models/model_local_v3.pt \
    --model2 models/model_multi.pt
```

### 3. **Text Reduction with Embeddings** (Partially Complete)
**Status**: Implemented but needs integration/testing

- ✅ `text_reduction_real.py` - ICFTextReducer with sentence-transformers
- ✅ `demo_text_reduction_real.py` - Demo script exists
- ❌ **Not Done**: Full integration testing
- ❌ **Not Done**: Evaluation metrics (regret, semantic similarity)
- ❌ **Not Done**: Comparison with baseline methods
- ❌ **Not Done**: Documentation and examples

**Next Steps**:
```bash
# Test text reduction
python scripts/demo_text_reduction_real.py \
    --model models/model_local_v3.pt \
    --text "the quick brown fox jumps over the lazy dog" \
    --target-ratio 0.5

# Evaluate on benchmark texts
# Create evaluation script for text reduction
```

## 🟡 Medium Priority - Enhancements

### 4. **Modern Words & Neologisms** (Partially Complete)
**Status**: Script exists but needs more data

- ✅ `add_modern_words.py` - Script to add modern words
- ✅ `word_frequency_modern.csv` - Created
- ❌ **Not Done**: Add more portmanteaus, neologisms
- ❌ **Not Done**: Test model performance on modern words
- ❌ **Not Done**: Evaluate generalization to new word types

**Next Steps**:
```bash
# Add more modern words
python scripts/add_modern_words.py \
    --input data/word_frequency.csv \
    --output data/word_frequency_modern_v2.csv \
    --add-portmanteaus \
    --add-neologisms

# Train on modern words and compare
```

### 5. **Historical Ngram Data** (Not Started)
**Status**: Script exists but not integrated

- ✅ `download_ngram_data.py` - Script exists
- ❌ **Not Done**: Download historical ngram data
- ❌ **Not Done**: Integrate into training pipeline
- ❌ **Not Done**: Test temporal generalization

**Next Steps**:
```bash
# Download ngram data
python scripts/download_ngram_data.py --output data/ngrams/

# Integrate into training
# Modify data loading to include historical frequencies
```

### 6. **Curriculum Learning** (Implemented, Needs Refinement)
**Status**: Works but may need tuning

- ✅ `train_curriculum.py` - Curriculum learning implemented
- ✅ `curriculum.py` - Curriculum sampler
- ❌ **Not Done**: Optimize curriculum stages
- ❌ **Not Done**: Compare curriculum vs standard training
- ❌ **Not Done**: Tune warmup epochs

**Next Steps**:
```bash
# Train with curriculum
python -m tiny_icf.train_curriculum \
    --data data/word_frequency.csv \
    --epochs 100 \
    --curriculum-stages 5 \
    --warmup-epochs 10

# Compare results
```

## 🟢 Lower Priority - Future Work

### 7. **Model Compression** (Not Started)
**Status**: Planned but not implemented

- ❌ Quantization (float16, int8)
- ❌ Pruning (remove low-importance weights)
- ❌ Knowledge distillation (small from large)
- ❌ Low-rank factorization

**References**: `MODEL_OPTIMIZATION_PLAN.md`

### 8. **Rust Deployment** (Partially Complete)
**Status**: Structure exists but needs integration

- ✅ `rust/` directory with basic structure
- ✅ `export_weights.py` - Weight export script
- ❌ **Not Done**: Integrate weight export into training
- ❌ **Not Done**: Test Rust inference
- ❌ **Not Done**: Benchmark Rust vs Python speed
- ❌ **Not Done**: Create Rust CLI tool

**Next Steps**:
```bash
# Export weights after training
python -m tiny_icf.export_weights \
    --model models/model_local_v3.pt \
    --output rust/weights.json

# Test Rust inference
cd rust && cargo run --release
```

### 9. **Advanced Evaluation** (Partially Complete)
**Status**: Basic eval exists, needs expansion

- ✅ `eval.py` - Comprehensive metrics
- ✅ `evaluate_model.py` - Evaluation script
- ❌ **Not Done**: Domain-specific evaluation (medical, legal, etc.)
- ❌ **Not Done**: Cross-lingual evaluation
- ❌ **Not Done**: Temporal evaluation (word frequency changes)
- ❌ **Not Done**: Error analysis (what words fail?)

**Next Steps**:
```bash
# Create domain-specific evaluation
# Create cross-lingual evaluation
# Create error analysis script
```

### 10. **Data Augmentation Improvements** (Partially Complete)
**Status**: Basic augmentation exists

- ✅ `augmentation.py` - Advanced augmentation
- ❌ **Not Done**: Test augmentation impact on performance
- ❌ **Not Done**: Tune augmentation probabilities
- ❌ **Not Done**: Add more augmentation strategies

## 📋 Summary by Category

### Architecture & Training
- [ ] Train HierarchicalICF variant
- [ ] Train BoxEmbeddingICF variant
- [ ] Train NanoICF variant
- [ ] Compare all architectures
- [ ] Test multi-loss training
- [ ] Optimize curriculum learning

### Data & Evaluation
- [ ] Add more modern words/neologisms
- [ ] Download and integrate historical ngrams
- [ ] Domain-specific evaluation
- [ ] Cross-lingual evaluation
- [ ] Error analysis

### Applications
- [ ] Complete text reduction integration
- [ ] Evaluate text reduction performance
- [ ] Create text reduction benchmarks

### Deployment
- [ ] Complete Rust integration
- [ ] Benchmark Rust inference
- [ ] Create production-ready Rust CLI

### Optimization
- [ ] Model quantization
- [ ] Model pruning
- [ ] Knowledge distillation
- [ ] Low-rank factorization

## Recommended Order

### Phase 1: Core Training (Current Focus)
1. ✅ Fix initialization and loss (DONE)
2. ✅ Add mid-training evaluation (DONE)
3. ⏳ Let current training finish
4. ⏳ Test multi-loss training
5. ⏳ Train architecture variants

### Phase 2: Data & Evaluation
1. Add more modern words
2. Download historical ngrams
3. Expand evaluation suite
4. Error analysis

### Phase 3: Applications
1. Complete text reduction
2. Evaluate text reduction
3. Create benchmarks

### Phase 4: Optimization & Deployment
1. Model compression
2. Rust deployment
3. Production optimization

## Quick Reference

**Train Architecture Variants**:
```bash
python scripts/train_variations.py --data data/word_frequency.csv --epochs 50
```

**Test Multi-Loss**:
```bash
python -m tiny_icf.train_multi_loss --data data/word_frequency.csv --multi-loss --epochs 100
```

**Test Text Reduction**:
```bash
python scripts/demo_text_reduction_real.py --model models/model_local_v3.pt
```

**Export for Rust**:
```bash
python -m tiny_icf.export_weights --model models/model_local_v3.pt --output rust/weights.json
```

