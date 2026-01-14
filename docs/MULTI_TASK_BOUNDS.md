# Theoretical Bounds for Multi-Task Output Features

## Overview

This document establishes theoretical bounds and expected ranges for multi-task learning outputs beyond ICF prediction:
- **Language Detection** (classification)
- **Era Classification** (classification)
- **Temporal ICF Prediction** (regression across decades)
- **Text Reduction** (embedding regret minimization)

These bounds help us understand whether multi-task learning is effective and if auxiliary tasks are helping or hurting ICF prediction.

## Notation

- **Acc**: Classification accuracy
- **L_CE**: Cross-entropy loss
- **L_MSE**: Mean squared error loss
- **ρ**: Spearman rank correlation coefficient
- **Regret**: Embedding regret (1 - cosine similarity)
- **W₂**: Wasserstein-2 distance (for optimal transport formulation)
- **I(S; Embedding)**: Mutual information between selected words S and embedding
- **H(Embedding)**: Shannon entropy of embedding distribution
- **N**: Number of classes (languages, eras)
- **k**: Number of words to keep (text reduction budget)

## 1. Language Detection

### Task Description

Predict the language of a word from character patterns. This is a multi-class classification task (typically 10+ languages).

### Theoretical Bound

**Formal Definition**:
\[
\text{Acc}_{\text{language}} = \frac{1}{|\mathcal{D}|} \sum_{(x, y) \in \mathcal{D}} \mathbb{1}[\hat{y}(x) = y]
\]
where \(\hat{y}(x)\) is the predicted language and \(y\) is the true language.

**Expected Range for Character-Level Models**:
- **Best case**: \(\text{Acc}_{\text{language}} \in [0.70, 0.85]\) (character patterns are language-specific)
- **Good**: \(\text{Acc}_{\text{language}} \in [0.60, 0.70]\)
- **Acceptable**: \(\text{Acc}_{\text{language}} \in [0.50, 0.60]\)
- **Poor**: \(\text{Acc}_{\text{language}} < 0.50\) (worse than random for balanced classes)

**Mathematical Foundation**:
- For \(N\) languages with balanced classes, random baseline: \(\text{Acc}_{\text{random}} = \frac{1}{N}\)
- For \(N=10\) languages: \(\text{Acc}_{\text{random}} = 0.10\)
- Character patterns (n-grams, character frequency) are language-specific
- Expected accuracy: \(\text{Acc}_{\text{language}} \in [0.60, 0.85]\) depending on language similarity
- Upper bound: \(\text{Acc}_{\text{language}} \leq 1 - \frac{H(\text{Language}|\text{Characters})}{H(\text{Language})}\) (Fano's inequality)

**Loss Components**:
- **Classification Loss** (Cross-Entropy): `0.2 - 0.5` (good), `> 0.7` (poor)
- **Ranking Loss**: `0.0 - 0.1` (good), `> 0.2` (poor)

**Interpretation**:
- Higher accuracy is better
- Language detection from characters alone is feasible (character patterns are language-specific)
- Values > 0.70 indicate strong language-specific patterns
- Values < 0.50 suggest model is not learning language features

**Connection to ICF**:
- Language detection helps ICF if languages have different frequency distributions
- Multi-task learning can improve ICF by learning language-specific features
- But if language accuracy is poor, it may hurt ICF prediction

## 2. Era Classification

### Task Description

Predict the historical era when a word was commonly used (e.g., 1800s, 1900s, 2000s). This is a multi-class classification task (typically 3-5 eras).

### Theoretical Bound

**Bound**: Depends on temporal character pattern changes

**Expected Range for Character-Level Models**:
- **Best case**: `0.50 - 0.70` accuracy (some temporal patterns exist)
- **Good**: `0.40 - 0.50` accuracy
- **Acceptable**: `0.30 - 0.40` accuracy
- **Poor**: `< 0.30` accuracy (worse than random for 3-5 classes)

**Mathematical Foundation**:
- For N eras with balanced classes, random baseline = 1/N
- For 5 eras: random = 0.20
- Character patterns change over time (spelling reforms, new words)
- But changes are subtle and may not be captured by character-level models
- Expected accuracy: `0.30 - 0.60` (lower than language detection)

**Loss Components**:
- **Classification Loss** (Cross-Entropy): `0.5 - 1.0` (good), `> 1.5` (poor)
- **Ranking Loss**: `0.0 - 0.1` (good), `> 0.2` (poor)

**Interpretation**:
- Higher accuracy is better
- Era detection from characters alone is challenging (temporal changes are subtle)
- Values > 0.50 indicate strong temporal patterns
- Values < 0.30 suggest model is not learning era features

**Connection to ICF**:
- Era detection helps ICF if word frequency changes over time
- Multi-task learning can improve ICF by learning temporal features
- But era detection is harder than language detection, so may be less helpful

## 3. Temporal ICF Prediction

### Task Description

Predict ICF scores across multiple decades (e.g., 1800, 1900, 2000). This is a regression task with temporal consistency constraints.

### Theoretical Bound

**Bound**: Similar to ICF prediction, but with temporal consistency

**Expected Range for Character-Level Models**:
- **Best case**: `0.15 - 0.18` Spearman per decade (similar to ICF)
- **Good**: `0.12 - 0.15` Spearman per decade
- **Acceptable**: `0.10 - 0.12` Spearman per decade
- **Poor**: `< 0.10` Spearman per decade

**Mathematical Foundation**:
- Each decade has its own ICF distribution
- Character patterns → ICF is still indirect (same bound as ICF)
- Temporal consistency helps: predictions should be smooth across decades
- Expected Spearman: `0.12 - 0.18` (similar to ICF, but may be slightly lower due to temporal complexity)

**Loss Components**:
- **Base Loss** (MSE): `0.05 - 0.10` (good), `> 0.20` (poor)
- **Consistency Loss**: `0.0 - 0.05` (good), `> 0.10` (poor)
- **Ranking Loss**: `0.05 - 0.15` (good), `> 0.30` (poor)

**Interpretation**:
- Higher Spearman is better
- Temporal ICF is harder than current ICF (predicting past/future)
- Consistency loss should be small (predictions should be smooth)
- Ranking loss ensures relative ordering across decades

**Connection to ICF**:
- Temporal prediction helps ICF by learning temporal patterns
- But temporal data may be sparse or noisy
- Multi-task learning can improve ICF if temporal patterns are consistent

## 4. Text Reduction (Embedding Regret Minimization)

### Task Description

Minimize embedding regret when reducing text by selecting a subset of words that preserves the original embedding as much as possible. This is a **ranking + embedding similarity** task that can be **disjoint from ICF prediction** (doesn't require ICF scores, but can use them as a heuristic).

**Key Insight**: The task is to find the minimal "path" of embedding regret - i.e., select words such that the embedding of the reduced text is as close as possible to the original embedding.

### Theoretical Bound

**Bound**: Depends on embedding quality, word selection strategy, and whether ICF is used

**Expected Range for Character-Level Models**:
- **Best case**: `0.05 - 0.15` regret (cosine distance)
- **Good**: `0.15 - 0.30` regret
- **Acceptable**: `0.30 - 0.50` regret
- **Poor**: `> 0.50` regret

**Mathematical Foundation**:
- **Regret** = 1 - cosine_similarity(original_embedding, reduced_embedding)
- **Path Regret**: Cumulative embedding change along the reduction path
- Perfect reduction (keeping all important words): regret ≈ 0.0
- Random reduction: regret ≈ 0.5 - 0.7
- **ICF-based reduction**: regret ≈ 0.15 - 0.30 (if ICF scores are accurate)
- **Direct embedding-based reduction** (disjoint from ICF): regret ≈ 0.10 - 0.25 (potentially better, as it directly optimizes embedding similarity)
- Expected regret: `0.15 - 0.30` for ICF-based, `0.10 - 0.25` for direct embedding-based

**Loss Components**:
- **Regret Loss** (Cosine Distance): `0.15 - 0.30` (good), `> 0.50` (poor)
- **Path Regret Loss** (if tracking cumulative changes): `0.20 - 0.40` (good), `> 0.60` (poor)
- **Ranking Loss**: `0.0 - 0.1` (good), `> 0.2` (poor)

**Interpretation**:
- Lower regret is better
- Regret < 0.15 indicates excellent text reduction
- Regret > 0.50 suggests reduction is not preserving meaning
- **Path regret** measures how much the embedding changes as words are removed incrementally
- Ranking loss ensures words are ranked correctly (by ICF or by embedding importance)

**Connection to ICF**:
- **Option 1 (Coupled)**: Text reduction uses ICF scores to rank words (rare words = important)
  - Better ICF prediction → better text reduction
  - Multi-task learning can improve both by learning shared features
- **Option 2 (Disjoint)**: Text reduction directly optimizes embedding similarity without ICF
  - Can be trained independently of ICF
  - May perform better (direct optimization vs proxy via ICF)
  - Still benefits from shared character-level features in multi-task setup

**Research Insight**:
- Direct embedding-based reduction (disjoint) may outperform ICF-based reduction
- ICF is a proxy for word importance, but embedding similarity is the actual objective
- Multi-task learning can learn both: ICF for ranking tasks, embedding regret for reduction tasks

## 5. Multi-Task Loss Weights

### Theoretical Bounds

**Task Weight Ratios**:
- **ICF / Total**: Should be dominant (0.5-0.8) since it's the primary task
- **Language / Total**: Should be moderate (0.1-0.2) since it's auxiliary
- **Era / Total**: Should be moderate (0.1-0.2) since it's auxiliary
- **Temporal / Total**: Should be moderate (0.1-0.3) since it's related to ICF
- **Text Reduction / Total**: Should be moderate (0.1-0.3) since it uses ICF

**AMOO (Aligned Multi-Objective Optimization)**:
- AMOO adaptively adjusts weights based on task difficulty
- Expected behavior: weights converge to balance task losses
- Warning sign: one task dominates (weight > 0.9) or is ignored (weight < 0.01)

## Summary Table

| Task | Metric | Best Case | Good | Acceptable | Poor | Interpretation |
|------|--------|-----------|------|------------|------|-----------------|
| **Language Detection** | Accuracy | 0.70-0.85 | 0.60-0.70 | 0.50-0.60 | < 0.50 | Higher is better |
| **Language Detection** | Classification Loss | 0.2-0.5 | 0.5-0.7 | 0.7-1.0 | > 1.0 | Lower is better |
| **Era Classification** | Accuracy | 0.50-0.70 | 0.40-0.50 | 0.30-0.40 | < 0.30 | Higher is better |
| **Era Classification** | Classification Loss | 0.5-1.0 | 1.0-1.5 | 1.5-2.0 | > 2.0 | Lower is better |
| **Temporal ICF** | Spearman | 0.15-0.18 | 0.12-0.15 | 0.10-0.12 | < 0.10 | Higher is better |
| **Temporal ICF** | Consistency Loss | 0.0-0.05 | 0.05-0.10 | 0.10-0.20 | > 0.20 | Lower is better |
| **Text Reduction** | Regret | 0.05-0.15 | 0.15-0.30 | 0.30-0.50 | > 0.50 | Lower is better |
| **Text Reduction** | Path Regret | 0.10-0.20 | 0.20-0.40 | 0.40-0.60 | > 0.60 | Lower is better |
| **Text Reduction** | Ranking Loss | 0.0-0.1 | 0.1-0.2 | 0.2-0.3 | > 0.3 | Lower is better |

## Practical Guidelines

### 1. Multi-Task Balance

Monitor task weight ratios:
- ICF should dominate (primary task)
- Auxiliary tasks should help, not hurt
- If auxiliary task accuracy is poor, reduce its weight

### 2. Convergence Indicators

- **Language accuracy**: Should increase to 0.60-0.70
- **Era accuracy**: Should increase to 0.40-0.50
- **Temporal Spearman**: Should increase to 0.12-0.15
- **Text reduction regret**: Should decrease to 0.15-0.30

### 3. Warning Signs

- **Language accuracy < 0.50**: Model not learning language features
- **Era accuracy < 0.30**: Model not learning era features
- **Temporal Spearman < 0.10**: Temporal prediction failing
- **Text reduction regret > 0.50**: Reduction not preserving meaning
- **One task dominates**: AMOO not balancing tasks

### 4. Multi-Task vs Single-Task

Compare:
- ICF Spearman in multi-task vs single-task
- If multi-task ICF < single-task ICF: auxiliary tasks may be hurting
- If multi-task ICF > single-task ICF: auxiliary tasks are helping

## Integration with Validation

These bounds should be:
1. **Logged during validation** for each task
2. **Compared to bounds** to identify issues
3. **Tracked over time** to monitor convergence
4. **Used for early stopping** if tasks don't improve

See `src/tiny_icf/flexible_lightning_module.py` for implementation.

