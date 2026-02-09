# Optimal Transport for Text Reduction: Summary

## Overview

Optimal Transport (OT) provides a principled framework for text reduction by modeling the problem as finding an optimal coupling between the original text's embedding distribution and the reduced text's embedding distribution.

## Formal Problem Statement

### Optimal Transport Formulation

Given:
- **Source distribution**: \(\mu_{\text{original}} = \frac{1}{n}\sum_{i=1}^n \delta_{\mathbf{e}_i}\) (empirical distribution over word embeddings)
- **Target distribution**: \(\mu_{\text{reduced}} = \frac{1}{k}\sum_{j=1}^k \delta_{\mathbf{e}'_j}\) (empirical distribution over selected word embeddings)
- **Cost matrix**: \(C_{ij} = \|\mathbf{e}_i - \mathbf{e}'_j\|^2\) (L2 distance) or \(C_{ij} = 1 - \cos(\mathbf{e}_i, \mathbf{e}'_j)\) (cosine distance)

Find the **transport plan** \(\gamma \in \Gamma(\mu_{\text{original}}, \mu_{\text{reduced}})\) that minimizes:
\[
W_2^2(\mu_{\text{original}}, \mu_{\text{reduced}}) = \min_{\gamma \in \Gamma} \sum_{i,j} \gamma_{ij} C_{ij}
\]
subject to marginal constraints:
\[
\sum_j \gamma_{ij} = \frac{1}{n}, \quad \sum_i \gamma_{ij} = \frac{1}{k}
\]

### Sinkhorn Algorithm

The **Sinkhorn algorithm** solves the entropy-regularized OT problem:
\[
W_2^{\varepsilon} = \min_{\gamma \in \Gamma} \left( \sum_{i,j} \gamma_{ij} C_{ij} + \varepsilon \cdot H(\gamma) \right)
\]
where \(H(\gamma) = -\sum_{i,j} \gamma_{ij} \log \gamma_{ij}\) is the entropy of the transport plan.

**Algorithm**:
1. Initialize: \(K_{ij} = \exp(-C_{ij}/\varepsilon)\)
2. Iterate until convergence:
   - \(u^{(t+1)} = \mathbf{1}_n / (K v^{(t)})\)
   - \(v^{(t+1)} = \mathbf{1}_k / (K^T u^{(t+1)})\)
3. Transport plan: \(\gamma_{ij} = u_i K_{ij} v_j\)

**Complexity**: O(n²·k·T) where T is number of iterations (typically 10-50)

### Wasserstein Distance as Regret

The **Wasserstein-2 distance** provides a principled regret metric:
\[
\text{Regret}_{\text{OT}} = W_2(\mu_{\text{original}}, \mu_{\text{reduced}}) = \left( \min_{\gamma \in \Gamma} \sum_{i,j} \gamma_{ij} C_{ij} \right)^{1/2}
\]

**Advantages**:
- **Geometric**: Measures distance in embedding space
- **Differentiable**: Sinkhorn provides smooth gradients
- **Optimal**: Finds globally optimal transport plan (for given ε)
- **Flexible**: Can handle soft assignments (fractional word selection)

**Connection to Cosine Regret**:
\[
\text{Regret}_{\text{cosine}} = 1 - \cos(\bar{\mathbf{e}}_{\text{original}}, \bar{\mathbf{e}}_{\text{reduced}})
\]
where \(\bar{\mathbf{e}}\) is the mean embedding.

For linear embeddings: \(W_2 \approx \text{Regret}_{\text{cosine}}\) (approximately)

## Comparison with Other Approaches

### ICF-Based Ranking

**Approach**: Rank words by ICF scores, keep top k
- **Complexity**: O(n log n) sorting
- **Theoretical**: No guarantee (ICF is proxy, not direct optimization)
- **Practical**: Fast, simple, often works well

### Embedding-Based Greedy

**Approach**: Greedily add word maximizing embedding similarity
- **Complexity**: O(n·k) for greedy
- **Theoretical**: (1-1/e) ≈ 0.63 approximation if submodular
- **Practical**: Good if embedding similarity is submodular

### Optimal Transport (Sinkhorn)

**Approach**: Find optimal transport plan via Sinkhorn
- **Complexity**: O(n²·k·T) where T ≈ 10-50 iterations
- **Theoretical**: Globally optimal (for given ε)
- **Practical**: Slower, but more stable gradients than NeuralSort

## Integration with rank-relax

**Current rank-relax methods**:
- "neural_sort": O(n log n), sharp rankings
- "probabilistic": O(n log n), smooth rankings
- "smooth_i": O(n log n), alternative gradient profile
- "sigmoid": O(n²), simple but slower

**Potential OT method**:
- "sinkhorn": O(n²·T), very stable, optimal transport-based
- Could be added as alternative ranking method
- Or used specifically for text reduction (embedding regret)

## Practical Recommendations

### When to Use OT

**Use OT/Sinkhorn when**:
- Need very stable gradients (OT is convex, smooth)
- Want globally optimal solution (for given ε)
- Embedding regret is the primary objective
- Can afford O(n²) complexity

**Use ICF-based when**:
- Need fast selection (O(n log n))
- ICF scores are accurate
- Simplicity is important

**Use greedy when**:
- Embedding similarity is submodular (or approximately)
- Want (1-1/e) approximation guarantee
- Need O(n·k) complexity

### Implementation Strategy

1. **Verify submodularity**: Test if embedding similarity is submodular
2. **Compare approaches**: ICF-based vs greedy vs OT
3. **Hybrid approach**: Use ICF for fast ranking, OT for fine-tuning
4. **Add to rank-relax**: Consider adding Sinkhorn as ranking method

## References

- Cuturi (2013): "Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances"
- Cuturi et al. (2019): "Differentiable Ranks and Sorting using Optimal Transport"
- Tang et al. (2022): "OTExtSum: Optimal Transport for Extractive Text Summarization"

