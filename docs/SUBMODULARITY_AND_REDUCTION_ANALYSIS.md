# Submodularity and Text Reduction Analysis

## Summary

This document summarizes the empirical verification of submodularity for embedding regret and the comparison of different text reduction approaches.

## Submodularity Verification

### Test Setup
- **Text**: "the quick brown fox jumps over the lazy dog" (9 words)
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Test Samples**: 50 random subset pairs (S ⊆ T)
- **Date**: 2025-01-06

### Results

| Metric | Value |
|--------|-------|
| **Violation Rate** | 16.0% (8/50) |
| **Avg Marginal Gain (Small Set S)** | 0.0402 |
| **Avg Marginal Gain (Large Set T)** | 0.0182 |
| **Is Submodular** | ❌ False |
| **Is Approximately Submodular** | ✅ True (< 20% violations) |

### Interpretation

**Submodularity Property**: For a function f to be submodular, it must satisfy:
```
f(S ∪ {v}) - f(S) ≥ f(T ∪ {v}) - f(T)  for all S ⊆ T
```

**Finding**: Embedding similarity (as measured by cosine similarity between average embeddings) is **approximately submodular** with a 16% violation rate.

**Implications**:
- ✅ **Greedy algorithms may work well** for embedding-based text reduction
- ⚠️ **No strict theoretical guarantee** (no (1-1/e) approximation guarantee)
- 💡 **Consider alternatives**: Optimal Transport, coreset algorithms, or other methods if strict guarantees are needed

### Theoretical Context

If embedding regret were perfectly submodular, greedy algorithms would have a (1-1/e) ≈ 0.63 approximation guarantee. Since it's only approximately submodular, we cannot rely on this guarantee, but empirical results suggest greedy methods may still perform well.

## Text Reduction Comparison

### Test Setup
- **Text**: "the quick brown fox jumps over the lazy dog" (9 words)
- **Target**: Keep 5 words (reduce from 9 to 5)
- **Embedding Model**: `all-MiniLM-L6-v2`
- **ICF Scores**: Synthetic (common words = 0.1, rare words = 0.5-0.9)

### Results

| Method | Selected Words | Regret | Improvement vs ICF |
|--------|---------------|--------|-------------------|
| **ICF-Based** | jumps, fox, lazy, quick, over | 0.472 | Baseline |
| **Embedding Greedy** | fox, jumps, lazy, dog, brown | 0.399 | **+15.4%** ✅ |
| **Hybrid** | fox, jumps, lazy, dog, brown | 0.399 | **+15.4%** ✅ |

**Best Method**: `embedding_greedy` (same as hybrid in this case)

### Key Findings

1. **Direct embedding optimization outperforms ICF-based reduction**
   - Greedy embedding optimization achieves 15.4% lower regret than ICF-based ranking
   - This suggests that directly optimizing embedding similarity is more effective than using ICF scores as a proxy

2. **Hybrid approach matches pure greedy**
   - ICF pre-filtering + embedding greedy achieves the same result as pure greedy
   - This suggests that ICF can be useful for initial candidate selection, but final selection should optimize embeddings directly

3. **Word selection differences**
   - ICF-based: Selected "jumps, fox, lazy, quick, over" (emphasizes rare words)
   - Embedding-based: Selected "fox, jumps, lazy, dog, brown" (better semantic preservation)

### Implications for Text Reduction Task

1. **For `TextReductionLoss`**: Consider using direct embedding optimization rather than relying solely on ICF scores
2. **For multi-task learning**: Text reduction can be **disjoint from ICF prediction** - it doesn't require ICF scores, but can use them as a heuristic
3. **For implementation**: The hybrid approach (ICF pre-filter + embedding greedy) may be a good compromise between speed and quality

## Recommendations

### For Text Reduction Implementation

1. **Primary Strategy**: Use embedding-based greedy optimization for text reduction
2. **Optional Enhancement**: Use ICF scores for initial candidate filtering (hybrid approach)
3. **Consider Optimal Transport**: For more principled optimization, consider using Sinkhorn algorithm for differentiable text reduction

### For Future Research

1. **Verify with real ICF scores**: Test with actual ICF predictions from trained models
2. **Test on longer texts**: Verify findings on longer documents (100+ words)
3. **Compare with Optimal Transport**: Implement Sinkhorn-based reduction and compare with greedy
4. **Submodularity on larger sets**: Test submodularity with more samples and larger word sets

## Scripts

- **`scripts/verify_submodularity.py`**: Verifies submodularity of embedding regret
- **`scripts/compare_reduction_approaches.py`**: Compares ICF-based, embedding-based, and hybrid reduction approaches

## References

- **Submodular Optimization**: Nemhauser et al. (1978) - Greedy algorithm for submodular maximization
- **Optimal Transport**: See `docs/OPTIMAL_TRANSPORT_SUMMARY.md` for OT-based text reduction
- **Text Reduction Theory**: See `docs/TEXT_REDUCTION_CS_CONNECTIONS.md` for connections to classic CS problems

