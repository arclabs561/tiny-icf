#set page(margin: 2cm)
#set text(size: 11pt)
#set heading(numbering: "1.")

// Code highlighting available via codly package if installed
// #import "@preview/codly:0.1.0": *
// #show: codly.with(theme: "github-dark")

#title[
  Technical Mathematical Description
]

#text(fill: gray)[
  ICF Estimation Project · #datetime.today().display()
]

#par(justify: true)[
  This document provides a complete technical mathematical description of the ICF prediction task, model architectures, loss functions, evaluation metrics, and training procedures.
]

== Task Formulation

=== Problem Definition

Given a corpus with word frequency counts, we aim to learn a function:

#align(center)[
  $ f: W -> [0, 1] $
]

that maps words to normalized Inverse Collection Frequency (ICF) scores, where:
- $W$ is the vocabulary space (words encoded as byte sequences)
- $f(w) in [0, 1]$ where $0$ = common word, $1$ = rare word

=== ICF Normalization

For a word $w$ with frequency count $c_w$ in a corpus of total tokens $T$:

#align(center)[
  $ y_w = frac(log(T + 1) - log(c_w + 1), log(T + 1)) $
]

where:
- Add-1 smoothing prevents edge cases (zero division, $c_w = T$)
- Result is clipped to $[0, 1]$: $y_w = "max"(0, "min"(1, y_w))$
- Words with $c_w < c_min$ (default: 5) are assigned $y_w = 1.0$ (treated as rare)

#strong[Properties:]

- #strong[Monotonic]: $c_(w_1) > c_(w_2) => y_(w_1) < y_(w_2)$
- #strong[Normalized]: $y_w in [0, 1]$ for all $w$
- #strong[Logarithmic scale]: captures Zipfian distribution

== Model Architectures

=== UniversalICF

#strong[Input:] Byte sequence $x in {0, ..., 255}^L$ (padded to max length $L$)

#strong[Forward Pass:]

1. #strong[Embedding:]

#align(center)[
  $ E = "Embedding"(x) in RR^(B times L times d_e) $
]

where $B$ = batch size, $d_e$ = embedding dimension (default: 36)

2. #strong[Convolutional Feature Extraction:]

For kernel sizes $k in {3, 5, 7}$:

#align(center)[
  $ C_k = "ReLU"("BatchNorm"("Conv1d"_k(E^T))) in RR^(B times d_c times L) $
]

where $d_c$ = conv channels (default: 18), $E^T$ transposes to $[B, d_e, L]$

3. #strong[Multi-Scale Pooling:]

For each $C_k$:
- Max pooling: $p_k^max = "max"_t C_k[., ., t] in RR^(B times d_c)$
- Mean pooling: $p_k^"mean" = frac(1, L) sum_t C_k[., ., t] in RR^(B times d_c)$
- Last position: $p_k^last = C_k[., ., -1] in RR^(B times d_c)$

4. #strong[Feature Concatenation:]

#align(center)[
  $ F = "concat"(p_3^max, p_3^"mean", p_3^last, p_5^max, p_5^"mean", p_5^last, p_7^max, p_7^"mean", p_7^last) in RR^(B times 9 d_c) $
]
]

5. #strong[MLP Head:]

#align(center)[
  $ h = "ReLU"("Dropout"("Linear"_1(F))) in RR^(B times d_h) $
]

#align(center)[
  $ hat(y) = "clamp"("Linear"_2(h), 0, 1) in RR^(B times 1) $
]

where $d_h$ = hidden dimension (default: 36), $"clamp"(x, a, b) = "max"(a, "min"(b, x))$

#strong[Total Parameters:] ~40k

=== ResidualICF

Similar to UniversalICF but with residual connection in MLP head:

#align(center)[
  $ h = "ReLU"("BatchNorm"("Linear"_1(F))) $
]

#align(center)[
  $ h_res = h + "Linear"_proj(F) $
]

#align(center)[
  $ hat(y) = "clamp"("Linear"_2("Dropout"(h_res)), 0, 1) $
]

where $"Linear"_proj$ projects $F$ to $d_h$ if dimensions don't match.

#strong[Total Parameters:] ~40k

== Loss Functions

=== Combined Loss

The total loss is a weighted combination:

#align(center)[
  $ L_total = L_huber + lambda_r L_rank + lambda_s L_spearman + lambda_n L_ndcg + lambda_l L_listwise $
]

where $lambda_r, lambda_s, lambda_n, lambda_l$ are component weights.

=== Huber Loss

Robust regression loss that behaves like MSE for small errors and MAE for large errors:

#align(center)[
  $ L_huber(hat(y), y) = frac(1, B) sum_(i=1)^B {
    frac(1, 2) (hat(y)_i - y_i)^2, "if" |hat(y)_i - y_i| < delta; \
    delta |hat(y)_i - y_i| - frac(1, 2) delta^2, "otherwise"
  } $
]

where $delta$ = threshold (default: 0.1). This prevents rare word outliers from exploding gradients.

=== Pairwise Ranking Loss

Enforces relative ordering: if word $w_i$ is more common than $w_j$ (i.e., $y_i < y_j$), then $hat(y)_i < hat(y)_j$.

#strong[Smooth Version (Sigmoid-based):]

#align(center)[
  $ L_rank(hat(y)_1, hat(y)_2, y_1, y_2) = frac(1, |P|) sum_((i,j) in P) sigma(tau(m - (hat(y)_j - hat(y)_i))) $
]

where:
- $P = {(i,j) : y_i < y_j, y_j - y_i >= m_min}$ is the set of valid pairs
- $m$ = margin (default: 0.1)
- $tau$ = temperature (default: 10.0)
- $sigma$ = sigmoid function
- Optionally weighted by target difference: $w_(ij) = "softmax"(alpha(y_j - y_i))$

#strong[Hard Version (ReLU-based):]

#align(center)[
  $ L_rank(hat(y)_1, hat(y)_2) = frac(1, |P|) sum_((i,j) in P) "max"(0, m - (hat(y)_j - hat(y)_i)) $
]

=== Spearman Correlation Loss

Directly optimizes Spearman rank correlation coefficient. Since ranking is non-differentiable, we use soft ranking approximations.

#strong[Soft Ranking (Vectorized):]

For predictions $hat(y) in RR^B$ and targets $y in RR^B$:

1. #strong[Compute soft ranks:]

#align(center)[
  $ hat(r)_i = frac(1, B-1) sum_(j != i) sigma(tau(hat(y)_i - hat(y)_j)) $
]

#align(center)[
  $ r_i = frac(1, B-1) sum_(j != i) sigma(tau(y_i - y_j)) $
]

where $tau$ = regularization strength (default: 0.1), $sigma$ = sigmoid

2. #strong[Center ranks:]

#align(center)[
  $ hat(r)'_i = hat(r)_i - bar(hat(r)), quad r'_i = r_i - bar(r) $
]

3. #strong[Compute Spearman correlation:]

#align(center)[
  $ rho_s = frac(sum_i hat(r)'_i r'_i, sqrt(sum_i (hat(r)'_i)^2) sqrt(sum_i (r'_i)^2) + epsilon) $
]

4. #strong[Loss:]

#align(center)[
  $ L_spearman = 1 - rho_s $
]

#strong[Alternative Backends:]

- #strong[torchsort]: Uses fast-soft-sort (O(n log n)) with compiled kernels
- #strong[diffsort]: Uses differentiable sorting networks (O(n²(log n)²))
- #strong[rank-relax]: Optimized Rust implementation with analytical gradients

=== NeuralNDCG Loss

Approximates Normalized Discounted Cumulative Gain:

1. #strong[Convert ICF to relevance:]

#align(center)[
  $ r_i = 1 - y_i $ (lower ICF = higher relevance)
]

2. #strong[Compute DCG:]

#align(center)[
  $ "DCG"@k = sum_(i=1)^k frac(r_(pi(i)), log_2(i + 1)) $
]

where $pi$ is the ranking induced by $hat(y)$ (descending)

3. #strong[Ideal DCG:]

#align(center)[
  $ "IDCG"@k = sum_(i=1)^k frac(r_(pi^*(i)), log_2(i + 1)) $
]

where $pi^*$ is the ideal ranking (by $r$ descending)

4. #strong[Loss:]

#align(center)[
  $ L_ndcg = 1 - frac("DCG"@k, "IDCG"@k) $
]

== Evaluation Metrics

=== Regression Metrics

#strong[Mean Absolute Error:]

#align(center)[
  $ "MAE" = frac(1, N) sum_(i=1)^N |hat(y)_i - y_i| $
]

#strong[Root Mean Squared Error:]

#align(center)[
  $ "RMSE" = sqrt(frac(1, N) sum_(i=1)^N (hat(y)_i - y_i)^2) $
]

#strong[Median Absolute Error:]

#align(center)[
  $ "MedAE" = "median"(|hat(y)_i - y_i|) $ (median function)
]

=== Correlation Metrics

#strong[Spearman Rank Correlation:]

#align(center)[
  $ rho_s = frac(sum_i (r_(hat(y),i) - bar(r)_hat(y)) (r_(y,i) - bar(r)_y), sqrt(sum_i (r_(hat(y),i) - bar(r)_hat(y))^2) sqrt(sum_i (r_(y,i) - bar(r)_y)^2)) $
]

where $r_(hat(y),i)$ and $r_(y,i)$ are ranks of $hat(y)_i$ and $y_i$ respectively.

#strong[Pearson Correlation:]

#align(center)[
  $ rho_p = frac(sum_i (hat(y)_i - bar(hat(y))) (y_i - bar(y)), sqrt(sum_i (hat(y)_i - bar(hat(y)))^2) sqrt(sum_i (y_i - bar(y))^2)) $
]

#strong[Kendall's Tau:]

#align(center)[
  $ tau = frac(2, N(N-1)) sum_(i<j) "sign"(hat(y)_i - hat(y)_j) cdot "sign"(y_i - y_j) $
]

=== Ranking Metrics

#strong[NDCG@k:]

#align(center)[
  $ "NDCG"@k = frac("DCG"@k, "IDCG"@k) $
]

where relevance $r_i = 1 - y_i$ (lower ICF = higher relevance).

#strong[Mean Average Precision (MAP):]

#align(center)[
  $ "MAP" = frac(1, |R|) sum_(r in R) "AP"(r) $
]

where $R$ is the set of relevant items and $"AP"(r)$ is average precision for item $r$.

#strong[Mean Reciprocal Rank (MRR):]

#align(center)[
  $ "MRR" = frac(1, |R|) sum_(r in R) frac(1, "rank"(r)) $
]

== Training Procedure

=== Optimization

#strong[Optimizer:] AdamW with component-specific learning rates:

- Embedding: $lr_e = lr cdot alpha_e$ (default: $alpha_e = 0.1$)
- Convolutional layers: $lr_c = lr cdot alpha_c$ (default: $alpha_c = 1.0$)
- MLP head: $lr_h = lr cdot alpha_h$ (default: $alpha_h = 1.0$)

#strong[Learning Rate Schedules:]

1. #strong[ReduceLROnPlateau:]

#align(center)[
  $ lr_(t+1) = {
    lr_t cdot gamma, "if" "val_spearman"_t "not improved for" p "epochs"; \
    lr_t, "otherwise"
  } $
]

where $gamma$ = factor (default: 0.5), $p$ = patience (default: 8-12)

2. #strong[Cosine Annealing with Warmup:]

#align(center)[
  $ lr_t = {
    lr_0 cdot frac(t, T_w), "if" t < T_w; \
    lr_0 cdot frac(1 + cos(pi frac(t - T_w, T - T_w)), 2), "if" t >= T_w
  } $
]

where $T_w$ = warmup epochs (default: 5), $T$ = total epochs

=== Regularization

#strong[Gradient Clipping:]

#align(center)[
  $ "clip"(bold(g), c) = bold(g) cdot "min"(1, frac(c, ||bold(g)||)) $
]

where $c$ = max norm (default: 1.0)

#strong[Dropout:] Applied in MLP head with probability $p$ (default: 0.3-0.4)

#strong[Weight Decay:] L2 regularization with coefficient $lambda_wd$ (default: $10^-4$)

=== Early Stopping

Stop training if validation Spearman correlation doesn't improve for $p$ epochs (default: 8-12) with minimum improvement $delta$ (default: 0.0001).

== Notation Summary

#table(
  columns: 2,
  align: left,
  stroke: 0.5pt,
  [*Symbol*], [*Description*],
  [$w$], [Word],
  [$c_w$], [Word frequency count],
  [$T$], [Total tokens in corpus],
  [$y_w$], [Normalized ICF score for word $w$],
  [$hat(y)$], [Model prediction],
  [$B$], [Batch size],
  [$L$], [Maximum sequence length],
  [$d_e$], [Embedding dimension],
  [$d_c$], [Convolutional channels],
  [$d_h$], [Hidden dimension],
  [$lambda_r, lambda_s, lambda_n, lambda_l$], [Loss component weights],
  [$delta$], [Huber loss threshold],
  [$m$], [Ranking margin],
  [$tau$], [Temperature/regularization strength],
  [$sigma$], [Sigmoid function],
  [$rho_s, rho_p, tau$], [Spearman, Pearson, Kendall correlations],
  [$lr$], [Learning rate],
  [$gamma$], [Learning rate decay factor],
  [$p$], [Patience (early stopping)],
  [$alpha_e, alpha_c, alpha_h$], [Component-specific LR multipliers],
)

