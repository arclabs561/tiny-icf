# Implementation Plan: Universal Frequency Estimator

## Overview

This document outlines the refined implementation plan for building a compressed, character-level model that predicts normalized ICF (Inverse Collection Frequency) for arbitrary words, satisfying Kolmogorov complexity constraints.

## Core Objective

Build a model where `K(Model) < K(Dictionary)` by learning the statistical structure of word commonality from character-level features rather than memorizing a lookup table.

## Architecture Decisions

### Why Byte-Level CNN?

1. **Compression**: Forces model to learn from parts, not whole words
2. **Generalization**: Handles typos, morphology, and unseen words
3. **Efficiency**: Parallel processing, fits in L2 cache (<50k params)
4. **Morphological Awareness**: Parallel kernels (3, 5, 7) capture prefixes, roots, suffixes

### Why Normalized ICF?

- **Bounded Output**: 0.0 (common) to 1.0 (rare) enables consistent thresholding
- **Universal Prior**: Works across domains without fine-tuning
- **Zero-Shot Utility**: Acts as informativeness weight for retrieval/classification

## Implementation Phases

### ✅ Phase 1: Data Construction (COMPLETE)

- [x] `load_frequency_list()`: CSV parser for word,count pairs
- [x] `compute_normalized_icf()`: ICF normalization formula
- [x] `stratified_sample()`: Zipfian-aware sampling (head/body/tail)
- [x] `WordICFDataset`: PyTorch dataset with augmentation

**Key Formula**:
```
ICF = (log(Total_Tokens) - log(Count)) / log(Total_Tokens)
```

### ✅ Phase 2: Model Architecture (COMPLETE)

- [x] `UniversalICF`: Byte-level CNN model
  - Embedding: 256 → 64
  - Parallel CNNs: kernel sizes 3, 5, 7
  - Global max pooling
  - MLP head with sigmoid

**Parameter Count**: ~45k (verified < 50k constraint)

### ✅ Phase 3: Training Infrastructure (COMPLETE)

- [x] `CombinedLoss`: Huber + Ranking loss
- [x] `train.py`: Full training loop with validation
- [x] Data augmentation: Character dropout/swap
- [x] Model checkpointing

**Loss Function**:
```
L = Huber(pred, target) + λ * Ranking(pred_pairs)
```

### ✅ Phase 4: Validation Framework (COMPLETE)

- [x] `test_jabberwocky.py`: Jabberwocky Protocol
- [x] `predict.py`: Inference CLI

**Jabberwocky Test Cases**:
- `"the"` → ~0.0 (common, memorized)
- `"xylophone"` → ~0.9 (rare but valid)
- `"flimjam"` → ~0.6-0.8 (rare, looks English)
- `"qzxbjk"` → ~0.99 (impossible structure)
- `"unfriendliness"` → ~0.4-0.6 (composed of common parts)

## Training Strategy

### Data Sampling

**Stratified Sampling** (handles Zipfian distribution):
- Head (top 10k): 40% of samples
- Body (10k-100k): 30% of samples
- Tail (100k+): 30% of samples

### Augmentation

**Character-Level Robustness**:
- 10% probability: Character dropout
- 10% probability: Adjacent character swap
- Target remains original word's ICF (forces structural learning)

### Loss Design

1. **Huber Loss**: Smooth L1, prevents rare word outliers from exploding gradients
2. **Ranking Loss**: Enforces relative ordering (common < rare in ICF space)

## Validation Criteria

### Success Metrics

1. **Jabberwocky Protocol**: Model correctly predicts scores for non-existent words
2. **Spearman Correlation**: > 0.8 on held-out words (proves structural learning)
3. **Parameter Count**: < 50k (compression constraint)
4. **Inference Speed**: < 1ms per word on CPU

### Failure Modes to Avoid

1. **Memorization**: Model predicts 0.0 for all training words, 0.5 for all others
2. **Length Bias**: Model only learns "shorter = more common"
3. **Temporal Overfitting**: Model memorizes topical spikes (e.g., "COVID")

## Next Steps

1. **Obtain Frequency Data**: Download Google 1T or similar large corpus
2. **Initial Training**: Train on subset (1M words) to verify pipeline
3. **Full Training**: Scale to full dataset (10M+ words)
4. **Validation**: Run Jabberwocky Protocol, measure correlation
5. **Integration**: Use as zero-shot prior in RAG/retrieval systems

## Research Questions

1. **Does character-level model beat lookup table?**
   - Measure: Model size vs. accuracy trade-off
   - Hypothesis: Model generalizes better on typos/morphology

2. **How does normalization affect zero-shot transfer?**
   - Measure: Retrieval accuracy with/without ICF weighting
   - Hypothesis: Normalized ICF improves rare word retrieval

3. **Can model learn beyond frequency?**
   - Measure: Correlation with concreteness, POS, etc.
   - Hypothesis: Character patterns encode semantic priors

## References

- Perplexity research summary (provided by user)
- SPLADE: Learned sparse representations
- Kolmogorov complexity constraints
- Zipfian distribution handling

