# Text Reduction: Connections to Classic CS Problems

## Overview

Text reduction (minimizing embedding regret while selecting a subset of words) is closely related to several classic computer science problems, particularly in **image processing**, **optimization**, and **information theory**. This document explores these connections and their implications for our approach.

## 1. Image Reduction/Compression Problems

### Parallel Problems

**Image Summarization**:
- **Problem**: Select key frames/images that represent the whole collection
- **Objective**: Minimize information loss while reducing size
- **Methods**: Submodular optimization, clustering, representative selection
- **Connection**: Text reduction is the same problem but for words instead of images

**Image Compression**:
- **Problem**: Reduce image size while preserving visual quality
- **Objective**: Minimize distortion (e.g., MSE, SSIM) for given bit budget
- **Methods**: Transform coding (DCT, wavelets), quantization, entropy coding
- **Connection**: Text reduction minimizes embedding distortion (cosine distance) for given word budget

**Image Downsampling**:
- **Problem**: Reduce image resolution while preserving important features
- **Objective**: Keep most informative pixels/regions
- **Methods**: Adaptive sampling, importance-based selection
- **Connection**: Text reduction keeps most informative words

### Key Insight

Both image and text reduction share the same **fundamental problem**:
- **Input**: High-dimensional representation (image pixels, text words)
- **Output**: Reduced representation (fewer pixels, fewer words)
- **Objective**: Minimize information loss (distortion, regret)
- **Constraint**: Budget (bit rate, word count)

## 2. Submodular Optimization

### Problem Formulation

**Submodular Function Maximization**:
- **Problem**: Select subset S ⊆ V that maximizes f(S) where f is submodular
- **Submodularity**: f(S ∪ {v}) - f(S) ≥ f(T ∪ {v}) - f(T) for S ⊆ T
- **Interpretation**: Diminishing returns (adding v to larger set gives less benefit)

**Text Reduction as Submodular Optimization**:
- **Set V**: All words in text
- **Set S**: Selected words (subset to keep)
- **Function f(S)**: Information preserved (e.g., embedding similarity to original)
- **Objective**: Maximize f(S) subject to |S| ≤ k (word budget)

### Research Findings

**Lin & Bilmes (2011)**: "A Class of Submodular Functions for Document Summarization"
- Shows that many summarization objectives are submodular
- Greedy algorithm achieves (1-1/e) ≈ 0.63 approximation guarantee
- Applies to sentence/word selection for summarization

**Mirzasoleiman et al. (2013)**: "Lazy Greedy Submodular Maximization"
- Efficient greedy algorithm for large-scale submodular optimization
- Reduces computational cost while maintaining approximation guarantee

**Connection to Our Problem**:
- If embedding similarity is submodular (or approximately submodular), we can use greedy algorithms
- Greedy selection: iteratively add word that maximizes embedding similarity
- Theoretical guarantee: (1-1/e) of optimal for submodular functions

## 3. Core Set Selection

### Problem Formulation

**Core Set Selection**:
- **Problem**: Find small subset that "represents" the whole dataset
- **Objective**: Minimize representation error (e.g., clustering cost, information loss)
- **Applications**: Active learning, dataset distillation, clustering

**Text Reduction as Core Set Selection**:
- **Dataset**: All words in text
- **Core Set**: Selected words (subset to keep)
- **Representation Error**: Embedding regret (distance from original embedding)
- **Objective**: Minimize regret for given core set size

### Research Findings

**Bachem et al. (2017)**: "Scalable k-Means Clustering via Lightweight Coresets"
- Shows how to find small coresets for clustering
- Coreset preserves clustering cost within (1+ε) factor
- Applies to any problem with clustering-like structure

**Connection to Our Problem**:
- Text reduction is similar: find "core words" that preserve embedding
- Can use coreset algorithms if embedding regret has clustering-like structure
- Theoretical guarantees: (1+ε) approximation for coreset-based selection

## 4. Knapsack Problem

### Problem Formulation

**0/1 Knapsack**:
- **Problem**: Select items with weights and values to maximize value within weight budget
- **Items**: Words
- **Weight**: Word count (or embedding space)
- **Value**: Information preserved (embedding similarity)
- **Budget**: Target word count

**Text Reduction as Knapsack**:
- **Items**: Words
- **Weight**: 1 (each word counts equally)
- **Value**: Contribution to embedding similarity (may depend on other selected words)
- **Budget**: Target word count k
- **Challenge**: Values are not independent (embedding is non-linear combination)

### Research Findings

**Greedy Knapsack**:
- If values are independent: greedy by value/weight ratio is optimal
- If values are dependent: greedy may not be optimal, but often good in practice
- For text reduction: values are dependent (embedding is sum/average of word embeddings)

**Connection to Our Problem**:
- Text reduction is a **dependent knapsack** problem
- Values depend on which other words are selected
- Greedy selection may work well if embedding is approximately linear
- Can use dynamic programming for small instances, greedy for large

## 5. Set Cover Problem

### Problem Formulation

**Set Cover**:
- **Problem**: Select minimum number of sets that cover all elements
- **Sets**: Words (each word "covers" some semantic concepts)
- **Elements**: Semantic concepts/information in original text
- **Objective**: Cover all concepts with minimum words

**Text Reduction as Set Cover**:
- **Sets**: Words (each word contributes to embedding)
- **Elements**: Dimensions of embedding space (or semantic concepts)
- **Objective**: Cover embedding space with minimum words
- **Challenge**: Words may overlap in coverage (embedding dimensions)

### Research Findings

**Greedy Set Cover**:
- Greedy algorithm: iteratively add set covering most uncovered elements
- Approximation guarantee: ln(n) + 1 for n elements
- Often performs well in practice

**Connection to Our Problem**:
- If embedding dimensions are "concepts" to cover, set cover applies
- Greedy selection: add word covering most uncovered embedding dimensions
- Theoretical guarantee: ln(d) + 1 for d embedding dimensions

## 6. Information-Theoretic Approaches

### Problem Formulation

**Information Maximization**:
- **Problem**: Select subset that maximizes mutual information I(S; Original)
- **Objective**: Maximize information preserved about original text
- **Constraint**: |S| ≤ k (word budget)

**Text Reduction as Information Maximization**:
- **S**: Selected words
- **Original**: Original text/embedding
- **I(S; Original)**: Mutual information between selected words and original
- **Objective**: Maximize I(S; Original) for given k

### Research Findings

**Information-Theoretic Bounds**:
- **Upper bound**: I(S; Original) ≤ H(Original) (entropy of original)
- **Achievable**: I(S; Original) ≈ H(Original) if S is sufficient statistic
- **For embeddings**: I(S; Original) ≤ H(Embedding) (entropy of embedding)

**Connection to Our Problem**:
- Embedding regret is related to information loss: Regret ≈ 1 - I(S; Original) / H(Original)
- Lower regret → higher mutual information
- Can use information-theoretic bounds to estimate achievable regret

## 8. Submodularity of Embedding Regret

### Key Question

**Is embedding regret submodular?**

**Definition**: f(S) = similarity(embedding(S), original_embedding) is submodular if:
- f(S ∪ {w}) - f(S) ≥ f(T ∪ {w}) - f(T) for S ⊆ T
- Interpretation: Adding word w to smaller set gives more benefit

**For Embeddings**:
- If embedding is **linear** (sum/average): f(S) = similarity(Σ_{w∈S} emb(w), orig)
  - Then f is **modular** (additive): f(S ∪ {w}) = f(S) + f({w})
  - Modular functions are submodular (special case)
- If embedding is **non-linear** (e.g., transformer): f may not be submodular
  - But often **approximately submodular** in practice

**Implications**:
- If submodular: greedy algorithm has (1-1/e) guarantee
- If approximately submodular: greedy often works well
- Can verify submodularity empirically or theoretically

## 9. Practical Algorithms

### Greedy Selection (Submodular Maximization)

```python
def greedy_text_reduction(words, embeddings, original_embedding, k):
    """
    Greedy selection: iteratively add word maximizing embedding similarity.
    
    If embedding similarity is submodular, this achieves (1-1/e) ≈ 0.63
    approximation guarantee.
    """
    selected = []
    remaining = list(range(len(words)))
    
    for _ in range(k):
        best_word = None
        best_similarity = -1
        
        for word_idx in remaining:
            candidate = selected + [word_idx]
            candidate_embedding = average([embeddings[i] for i in candidate])
            similarity = cosine_similarity(candidate_embedding, original_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_word = word_idx
        
        selected.append(best_word)
        remaining.remove(best_word)
    
    return selected
```

### Lazy Greedy (Efficient)

```python
def lazy_greedy_text_reduction(words, embeddings, original_embedding, k):
    """
    Lazy greedy: maintain priority queue of marginal gains.
    
    More efficient than standard greedy for large vocabularies.
    """
    # Implementation uses priority queue to avoid recomputing all similarities
    # See Mirzasoleiman et al. (2013) for details
    pass
```

### Coreset-Based Selection

```python
def coreset_text_reduction(words, embeddings, original_embedding, k, epsilon=0.1):
    """
    Coreset selection: find small subset preserving embedding within (1+ε) factor.
    
    Theoretical guarantee: regret ≤ (1+ε) * optimal_regret
    """
    # Use coreset algorithms (e.g., k-means coreset) adapted for embeddings
    # See Bachem et al. (2017) for details
    pass
```

## 10. Theoretical Bounds for Text Reduction

### Information-Theoretic Lower Bound

**Regret Lower Bound**:
- **H(Embedding)**: Entropy of original embedding
- **H(Embedding | S)**: Entropy of embedding given selected words
- **Regret ≥ 1 - I(S; Embedding) / H(Embedding)**
- **I(S; Embedding)**: Mutual information between selected words and embedding

**For k words**:
- Maximum I(S; Embedding) ≤ k * H(word) (if words are independent)
- But words are dependent, so I(S; Embedding) ≤ H(Embedding) (data processing inequality)
- **Lower bound**: Regret ≥ 1 - min(k * H(word), H(Embedding)) / H(Embedding)

### Submodular Maximization Upper Bound

**If embedding similarity is submodular**:
- Greedy algorithm achieves \((1-1/e) \approx 0.63\) of optimal
- **Upper bound**: 
  \[
  \text{Regret}_{\text{greedy}} \leq \left(1 - \left(1-\frac{1}{e}\right)\right) \cdot \text{Regret}_{\text{optimal}} + \frac{1}{e} \cdot \text{Regret}_{\text{worst}}
  \]
- In practice: \(\text{Regret}_{\text{greedy}} \approx 0.37 \cdot \text{Regret}_{\text{optimal}} + 0.63 \cdot \text{Regret}_{\text{worst}}\)

**For typical text reduction**:
- Optimal regret: \(\text{Regret}_{\text{optimal}} \approx 0.10-0.15\) (if we could solve exactly)
- Greedy regret: \(\text{Regret}_{\text{greedy}} \approx 0.15-0.30\) (using greedy approximation)
- Worst case regret: \(\text{Regret}_{\text{worst}} \approx 0.50-0.70\) (random selection)

### Coreset Upper Bound

**If using coreset algorithms**:
\[
\text{Regret}_{\text{coreset}} \leq (1+\varepsilon) \cdot \text{Regret}_{\text{optimal}}
\]
where \(\varepsilon > 0\) is the approximation parameter (typically \(\varepsilon \in [0.1, 0.5]\)).

## 11. Integration with ICF Prediction

### Coupled Approach (ICF-Based Ranking)

**Advantages**:
- ICF scores provide word importance (rare words = informative)
- Simple: rank by ICF, keep top k
- Fast: O(n log n) sorting

**Disadvantages**:
- ICF is proxy, not direct optimization of embedding regret
- May not be optimal for embedding preservation
- Assumes ICF correlates with embedding importance

### Disjoint Approach (Direct Embedding Optimization)

**Advantages**:
- Directly optimizes embedding regret (actual objective)
- Can use submodular/greedy algorithms with theoretical guarantees
- May outperform ICF-based selection

**Disadvantages**:
- More expensive: O(n * k) for greedy, O(n²) for optimal
- Requires embedding model (additional dependency)
- May not generalize to other tasks (ICF is more general)

### Hybrid Approach (Multi-Task)

**Best of Both Worlds**:
- Train ICF prediction (general word importance)
- Train embedding regret minimization (task-specific optimization)
- Multi-task learning: shared features help both tasks
- Use ICF for fast ranking, embedding regret for fine-tuning

## 12. Practical Recommendations

### For Our Implementation

1. **Verify Submodularity**:
   - Test if embedding similarity is submodular (or approximately)
   - If yes: use greedy with (1-1/e) guarantee
   - If no: still try greedy (often works well)

2. **Compare Approaches**:
   - ICF-based: Fast, simple, general
   - Embedding-based: Direct optimization, potentially better
   - Hybrid: Best of both, multi-task learning

3. **Theoretical Bounds**:
   - Information-theoretic lower bound: Regret ≥ 1 - I(S; Embedding) / H(Embedding)
   - Submodular upper bound: Regret ≤ 0.37 * optimal + 0.63 * worst_case
   - Use these to evaluate algorithm performance

4. **Path Regret Tracking**:
   - Track cumulative embedding changes along reduction path
   - Helps understand if reduction is smooth (isotonic) or jagged
   - Can optimize for smooth paths (minimize path regret)

## 13. Research Directions

### Future Work

1. **Submodularity Verification**:
   - Prove or disprove submodularity of embedding regret
   - If not submodular, find approximation guarantees

2. **Coreset Algorithms**:
   - Adapt coreset algorithms for text reduction
   - Theoretical guarantees: (1+ε) approximation

3. **Information-Theoretic Bounds**:
   - Derive tighter bounds on achievable regret
   - Connect to Kolmogorov complexity (like ICF bounds)

4. **Hybrid Optimization**:
   - Combine ICF ranking with embedding optimization
   - Multi-objective: ICF importance + embedding regret

5. **Path Optimization**:
   - Optimize for smooth reduction paths (isotonic regret)
   - May improve final regret by avoiding local minima

## Summary

Text reduction is fundamentally the same problem as:
- **Image summarization** (select representative items)
- **Submodular maximization** (select subset maximizing function)
- **Core set selection** (find small representative subset)
- **Knapsack problem** (select items within budget)
- **Set cover** (cover all concepts with minimum sets)
- **Information maximization** (maximize mutual information)

**Key Insight**: If embedding similarity is submodular (or approximately), greedy algorithms have theoretical guarantees and often work well in practice.

**Practical Approach**: 
- Test submodularity empirically
- Compare ICF-based vs embedding-based vs hybrid
- Use theoretical bounds to evaluate performance
- Track path regret for smooth reductions

