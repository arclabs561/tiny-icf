#set page(margin: 2cm)
#set text(size: 11pt)
#set heading(numbering: "1.")

// Code highlighting available via codly package if installed
// #import "@preview/codly:0.1.0": *
// #show: codly.with(theme: "github-dark")

#title[
  Performance Ceiling Analysis: Why ~0.18-0.19 Spearman Correlation?
]

#text(fill: gray)[
  ICF Estimation Project · #datetime.today().display()
]

#par(justify: true)[
  This document analyzes the observed performance ceiling of ~0.18-0.19 Spearman correlation, attributing it to information-theoretic bounds, Kolmogorov complexity, architectural limitations, and loss-metric mismatch.
]

== Notation

- #strong[ρ]: Spearman rank correlation coefficient
- #strong[H(ICF)]: Shannon entropy of ICF distribution
- #strong[I(Characters; ICF)]: Mutual information between character patterns and ICF
- #strong[K(x)]: Kolmogorov complexity of x (length of shortest program producing x)
- #strong[K(M)]: Kolmogorov complexity of model M
- #strong[K(D)]: Kolmogorov complexity of dictionary D (word → ICF mapping)
- #strong[σ²]: Variance
- #strong[E[·]]: Expected value

== Observed Performance

#strong[Observed Results:]

- #strong[Best result:] loss_ablation_balanced_hybrid (0.1891)
- #strong[Iter4 distillation:] 0.1875
- #strong[Residual balanced:] 0.1864
- #strong[Consistent ceiling:] ~0.18-0.19 across multiple experiments

This suggests a #strong[fundamental limit], not just an optimization issue.

== Theoretical Limits

=== Information-Theoretic Bound

Maximum Spearman correlation is bounded by:

#align(center)[
  $ rho_max <= sqrt(frac(I(X; Y), H(Y))) $
]

Where:
- #strong[X] = Character features
- #strong[Y] = ICF values
- #strong[I(X; Y)] = Mutual information between characters and ICF
- #strong[H(Y)] = Shannon entropy of ICF distribution

=== Formal Derivation

#align(center)[
  $ rho_max^2 <= frac(I(X; Y), H(Y)) <= frac(H(X) - H(X|Y), H(Y)) $
]

This follows from the data processing inequality and the relationship between correlation and mutual information.

=== Key Insight

If `I(Characters; ICF)` is low relative to `H(ICF)`, the maximum achievable correlation is inherently limited.

#let hypothesis = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Hypothesis:] Character-level features capture approximately 18-19% of ICF variance because:
  - Character patterns → semantic frequency is an #strong[indirect mapping]
  - Many words with similar character patterns have different ICF values
  - ICF depends on corpus/domain characteristics (not just characters)
]

#hypothesis

=== Kolmogorov Complexity Bound

#strong[K(ICF | Characters)] measures the minimum information needed beyond character patterns to predict ICF.

If `K(ICF | Characters)` is large:
- Character patterns are insufficient
- Additional information (semantics, context) is required
- Performance ceiling is reached

== Architectural Limitations

=== Current Design Captures

- ✅ Character-level morphological patterns
- ✅ N-gram features (3, 5, 7 character windows)
- ✅ Word-level patterns (via pooling)

=== Missing Information

- ❌ Semantic understanding (word meaning)
- ❌ Document context (domain, type, style)
- ❌ Co-occurrence patterns (word relationships)
- ❌ Temporal/domain trends
- ❌ Corpus-specific characteristics

=== Why This Matters

ICF fundamentally depends on:
1. #strong[Word meaning] (semantic frequency)
2. #strong[Document context] (domain-specific usage)
3. #strong[Corpus characteristics] (training data distribution)

Character patterns alone cannot capture this information.

== Loss-Metric Mismatch

=== Problem

- #strong[Training objective:] MSE/Huber loss (minimizes absolute error)
- #strong[Evaluation metric:] Spearman correlation (measures ranking quality)

#strong[Problem:]
- Model optimizes for absolute accuracy
- But we care about relative ordering (ranking)
- Mismatch causes suboptimal optimization

#strong[Solution:] Direct Spearman optimization (already implemented via `rank-relax`)

== Data Quality & Noise

- ICF computed from specific corpus (bias)
- Measurement noise in frequency counts
- Domain mismatch between train/test
- Limited training data

== Why 0.18-0.19 Specifically?

=== Hypothesis

Character features capture ~18-19% of ICF variance because:

1. #strong[Information content:] `I(Characters; ICF) / H(ICF) ≈ 0.18-0.19`
   - Characters provide limited information about frequency
   - Semantic/contextual information is missing

2. #strong[Mapping complexity:] Character patterns → ICF is:
   - #strong[Many-to-one:] Different words → same ICF
   - #strong[One-to-many:] Similar patterns → different ICF
   - #strong[Ambiguous:] Requires additional information

3. #strong[Architectural limits:] Current CNN design:
   - No semantic understanding
   - No document context
   - Limited receptive field (word length only)

== Breaking the Ceiling

=== Option 1: Add Semantic Features

- Word embeddings (capture meaning)
- Pre-trained language model features
- Semantic similarity to known words

=== Option 2: Add Context

- Document type/domain
- Co-occurrence patterns
- Temporal trends

=== Option 3: Larger Architecture

- Attention mechanisms (long-range dependencies)
- Transformer-based (semantic understanding)
- Multi-scale features

=== Option 4: Direct Spearman Optimization

- Train directly on Spearman loss (already implemented)
- Better alignment with evaluation metric

=== Option 5: Multi-Task Learning

- Predict multiple related tasks
- Share semantic representations
- Improve ICF prediction

== Mathematical Formulation

=== Current

#align(center)[
  $ f: C^n -> [0, 1] $
]

Where `C` = character vocabulary, `n` = word length. Limited because `I(C^n; ICF)` is bounded.

=== Better

#align(center)[
  $ f: (C^n, S, D) -> [0, 1] $
]

Where `S` = semantic features, `D` = document context. This increases `I(Features; ICF)`.

== Conclusion

The ~0.18-0.19 ceiling is likely due to:

1. #strong[Information-theoretic limit:] Characters provide limited ICF information
2. #strong[Architectural limitations:] Missing semantic/contextual features
3. #strong[Task difficulty:] Character patterns → frequency is indirect
4. #strong[Loss-metric mismatch:] Suboptimal optimization (partially addressed)

To break the ceiling, we need to:

- Add semantic features (word embeddings, LM features)
- Add document context
- Use larger architectures with attention
- Continue direct Spearman optimization
- Explore multi-task learning

