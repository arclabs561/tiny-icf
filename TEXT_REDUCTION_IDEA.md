# Optimal Text Reduction Using ICF: Minimizing Embedding Regret

## The Core Idea

**Use ICF scores to guide optimal text reduction that minimizes embedding difference (regret).**

### Problem Statement

Given a text and a target length, find the optimal subset of words that:
1. Minimizes embedding difference (regret) from original
2. Preserves semantic information
3. Uses ICF to guide what to drop

### Key Insight

**Common words (low ICF) contribute less to semantic meaning than rare words (high ICF).**

Therefore, dropping common words first should minimize embedding regret while preserving informative content.

## How It Works

### 1. **Greedy ICF Reduction** (Simple)

**Algorithm:**
1. Compute ICF for each word
2. Sort words by ICF (ascending: common first)
3. Keep top N words (highest ICF = most informative)
4. Drop bottom words (lowest ICF = least informative)

**Example:**
```
Original: "the quick brown fox jumps over the lazy dog"
ICF:      [0.1, 0.5, 0.6, 0.4, 0.5, 0.3, 0.1, 0.4, 0.3]

Sorted:   "the" (0.1), "the" (0.1), "over" (0.3), "dog" (0.3),
          "fox" (0.4), "lazy" (0.4), "quick" (0.5), "jumps" (0.5),
          "brown" (0.6)

Keep top 5: "brown quick jumps fox lazy"
Dropped: "the the over dog" (low ICF)
```

**Regret:** Compute embedding difference between original and reduced text.

### 2. **Optimal Reduction with Regret** (Better)

**Algorithm:**
1. Compute original embedding
2. Iteratively try dropping each word
3. Compute embedding regret for each candidate
4. Drop word that causes least regret (weighted by ICF)
5. Repeat until target length

**Why Better:**
- Considers actual embedding impact, not just ICF
- Can handle cases where ICF doesn't perfectly predict importance
- More accurate regret minimization

### 3. **Beam Search** (Best, but slower)

**Algorithm:**
1. Maintain top-K candidates at each step
2. For each candidate, try dropping each word
3. Keep top-K new candidates (lowest regret)
4. Repeat until target length

**Why Best:**
- Explores multiple paths
- More likely to find global optimum
- But slower (O(K * N^2))

## Relationship to Embeddings

### The Connection

**ICF-guided reduction is related to embedding optimization:**

1. **Embedding Preservation**: Goal is to minimize embedding difference
2. **Information Content**: ICF measures word informativeness
3. **Optimal Selection**: Choose words that preserve semantics

### Why This Works

**Research shows:**
- Rare words (high ICF) carry more semantic information
- Common words (low ICF) are often redundant
- Embeddings are weighted by word importance
- Dropping low-ICF words preserves embedding structure

### Applications

1. **Text Summarization**: Reduce length while preserving meaning
2. **Token Filtering**: Before expensive embedding computation
3. **Compression**: Reduce text size with minimal semantic loss
4. **RAG Optimization**: Keep only informative tokens

## Implementation

### Basic Usage

```python
from tiny_icf.text_reduction import reduce_text_with_icf

text = "the quick brown fox jumps over the lazy dog"
reduced, regret, stats = reduce_text_with_icf(
    text=text,
    icf_model=icf_model,
    embedding_model=embedding_model,
    target_ratio=0.5,  # Keep 50% of words
    method="greedy",  # or "dynamic" or "beam"
)

print(f"Original: {text}")
print(f"Reduced: {reduced}")
print(f"Regret: {regret:.4f}")
print(f"Stats: {stats}")
```

### Integration with Embeddings

**For real embeddings (e.g., sentence-transformers):**

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embedding_fn(words: List[str]) -> torch.Tensor:
    text = ' '.join(words)
    return embedding_model.encode(text, convert_to_tensor=True)
```

## Optimization Strategies

### 1. **ICF-Weighted Regret**

Weight embedding regret by ICF:
```python
weighted_regret = regret * (1.0 - icf_score)
```

**Rationale:** Low ICF words should cause less regret when dropped.

### 2. **Progressive Reduction**

Reduce in stages:
1. Drop very common words (ICF < 0.2)
2. Drop common words (ICF < 0.4)
3. Drop moderate words (ICF < 0.6)
4. Keep rare words (ICF > 0.6)

### 3. **Position-Aware**

Consider word position:
- Keep important words at sentence boundaries
- Preserve sentence structure
- Weight by position + ICF

## Evaluation

### Metrics

1. **Embedding Regret**: Cosine distance between original/reduced embeddings
2. **ICF Preservation**: Average ICF of kept vs dropped words
3. **Semantic Similarity**: Human evaluation or downstream task performance
4. **Compression Ratio**: Words kept / words original

### Benchmarks

**Target Performance:**
- Regret < 0.1 for 50% reduction (good)
- Regret < 0.2 for 70% reduction (acceptable)
- ICF of kept words > ICF of dropped words (should be true)

## Research Questions

1. **Does ICF-guided reduction outperform random/tf-idf?**
   - Hypothesis: Yes, ICF better predicts semantic importance

2. **What's the optimal regret-ratio trade-off?**
   - How much can we reduce before regret explodes?

3. **Does this work across domains?**
   - General text vs domain-specific (medical, legal, etc.)

4. **How does it compare to learned summarization?**
   - ICF-guided vs neural summarization models

## Future Directions

1. **Learn Optimal Reduction Policy**
   - Train model to predict which words to drop
   - Use embedding regret as reward signal

2. **Multi-Objective Optimization**
   - Minimize regret
   - Maximize ICF of kept words
   - Preserve sentence structure

3. **Context-Aware Reduction**
   - Consider surrounding words
   - Preserve important phrases
   - Maintain coherence

4. **Real-Time Applications**
   - Fast greedy reduction for streaming text
   - Incremental regret computation
   - Cached embeddings

## Connection to Our Project

**This is a perfect downstream application of ICF estimation!**

1. **Use Case**: Text reduction before embedding computation
2. **ICF Model**: Predicts word informativeness
3. **Optimization**: Minimize embedding regret
4. **Evaluation**: Measure how well ICF guides reduction

**This validates the ICF model's utility beyond simple filtering.**

