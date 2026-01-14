# Technical Mathematical Description

## 1. Task Formulation

### 1.1 Problem Definition

Given a corpus with word frequency counts, we aim to learn a function $f: \mathcal{W} \to [0,1]$ that maps words to normalized Inverse Collection Frequency (ICF) scores, where:
- $\mathcal{W}$ is the vocabulary space (words encoded as byte sequences)
- $f(w) \in [0,1]$ where $0$ = common word, $1$ = rare word

### 1.2 ICF Normalization

For a word $w$ with frequency count $c_w$ in a corpus of total tokens $T$:

$$y_w = \frac{\log(T + 1) - \log(c_w + 1)}{\log(T + 1)}$$

where:
- Add-1 smoothing prevents edge cases (zero division, $c_w = T$)
- Result is clipped to $[0, 1]$: $y_w = \max(0, \min(1, y_w))$
- Words with $c_w < c_{\min}$ (default: 5) are assigned $y_w = 1.0$ (treated as rare)

**Properties:**
- Monotonic: $c_{w_1} > c_{w_2} \implies y_{w_1} < y_{w_2}$
- Normalized: $y_w \in [0, 1]$ for all $w$
- Logarithmic scale: captures Zipfian distribution

## 2. Model Architectures

### 2.1 UniversalICF

**Input:** Byte sequence $x \in \{0, \ldots, 255\}^L$ (padded to max length $L$)

**Forward Pass:**

1. **Embedding:**
   $$E = \text{Embedding}(x) \in \mathbb{R}^{B \times L \times d_e}$$
   where $B$ = batch size, $d_e$ = embedding dimension (default: 36)

2. **Convolutional Feature Extraction:**
   For kernel sizes $k \in \{3, 5, 7\}$:
   $$C_k = \text{ReLU}(\text{BatchNorm}(\text{Conv1d}_k(E^T))) \in \mathbb{R}^{B \times d_c \times L}$$
   where $d_c$ = conv channels (default: 18), $E^T$ transposes to $[B, d_e, L]$

3. **Multi-Scale Pooling:**
   For each $C_k$:
   - Max pooling: $p_k^{\max} = \max_{t} C_k[:, :, t] \in \mathbb{R}^{B \times d_c}$
   - Mean pooling: $p_k^{\text{mean}} = \frac{1}{L}\sum_t C_k[:, :, t] \in \mathbb{R}^{B \times d_c}$
   - Last position: $p_k^{\text{last}} = C_k[:, :, -1] \in \mathbb{R}^{B \times d_c}$

4. **Feature Concatenation:**
   $$F = \text{concat}(p_3^{\max}, p_3^{\text{mean}}, p_3^{\text{last}}, p_5^{\max}, p_5^{\text{mean}}, p_5^{\text{last}}, p_7^{\max}, p_7^{\text{mean}}, p_7^{\text{last}}) \in \mathbb{R}^{B \times 9d_c}$$

5. **MLP Head:**
   $$h = \text{ReLU}(\text{Dropout}(\text{Linear}_1(F))) \in \mathbb{R}^{B \times d_h}$$
   $$\hat{y} = \text{clamp}(\text{Linear}_2(h), 0, 1) \in \mathbb{R}^{B \times 1}$$
   where $d_h$ = hidden dimension (default: 36), $\text{clamp}(x, a, b) = \max(a, \min(b, x))$

**Total Parameters:** ~40k

### 2.2 ResidualICF

Similar to UniversalICF but with residual connection in MLP head:

$$h = \text{ReLU}(\text{BatchNorm}(\text{Linear}_1(F)))$$
$$h_{\text{res}} = h + \text{Linear}_{\text{proj}}(F)$$
$$\hat{y} = \text{clamp}(\text{Linear}_2(\text{Dropout}(h_{\text{res}})), 0, 1)$$

where $\text{Linear}_{\text{proj}}$ projects $F$ to $d_h$ if dimensions don't match.

**Total Parameters:** ~40k

## 3. Loss Functions

### 3.1 Combined Loss

The total loss is a weighted combination:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{huber}} + \lambda_r \mathcal{L}_{\text{rank}} + \lambda_s \mathcal{L}_{\text{spearman}} + \lambda_n \mathcal{L}_{\text{ndcg}} + \lambda_l \mathcal{L}_{\text{listwise}}$$

where $\lambda_r, \lambda_s, \lambda_n, \lambda_l$ are component weights.

### 3.2 Huber Loss

Robust regression loss that behaves like MSE for small errors and MAE for large errors:

$$\mathcal{L}_{\text{huber}}(\hat{y}, y) = \frac{1}{B}\sum_{i=1}^B \begin{cases}
\frac{1}{2}(\hat{y}_i - y_i)^2 & \text{if } |\hat{y}_i - y_i| < \delta \\
\delta|\hat{y}_i - y_i| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

where $\delta$ = threshold (default: 0.1). This prevents rare word outliers from exploding gradients.

### 3.3 Pairwise Ranking Loss

Enforces relative ordering: if word $w_i$ is more common than $w_j$ (i.e., $y_i < y_j$), then $\hat{y}_i < \hat{y}_j$.

**Smooth Version (Sigmoid-based):**
$$\mathcal{L}_{\text{rank}}(\hat{y}_1, \hat{y}_2, y_1, y_2) = \frac{1}{|\mathcal{P}|}\sum_{(i,j) \in \mathcal{P}} \sigma(\tau(m - (\hat{y}_j - \hat{y}_i)))$$

where:
- $\mathcal{P} = \{(i,j) : y_i < y_j, y_j - y_i \geq m_{\min}\}$ is the set of valid pairs
- $m$ = margin (default: 0.1)
- $\tau$ = temperature (default: 10.0)
- $\sigma$ = sigmoid function
- Optionally weighted by target difference: $w_{ij} = \text{softmax}(\alpha(y_j - y_i))$

**Hard Version (ReLU-based):**
$$\mathcal{L}_{\text{rank}}(\hat{y}_1, \hat{y}_2) = \frac{1}{|\mathcal{P}|}\sum_{(i,j) \in \mathcal{P}} \max(0, m - (\hat{y}_j - \hat{y}_i))$$

### 3.4 Spearman Correlation Loss

Directly optimizes Spearman rank correlation coefficient. Since ranking is non-differentiable, we use soft ranking approximations.

**Soft Ranking (Vectorized):**
For predictions $\hat{y} \in \mathbb{R}^B$ and targets $y \in \mathbb{R}^B$:

1. **Compute soft ranks:**
   $$\hat{r}_i = \frac{1}{B-1}\sum_{j \neq i} \sigma(\tau(\hat{y}_i - \hat{y}_j))$$
   $$r_i = \frac{1}{B-1}\sum_{j \neq i} \sigma(\tau(y_i - y_j))$$
   where $\tau$ = regularization strength (default: 0.1), $\sigma$ = sigmoid

2. **Center ranks:**
   $$\hat{r}'_i = \hat{r}_i - \bar{\hat{r}}, \quad r'_i = r_i - \bar{r}$$

3. **Compute Spearman correlation:**
   $$\rho_s = \frac{\sum_i \hat{r}'_i r'_i}{\sqrt{\sum_i (\hat{r}'_i)^2} \sqrt{\sum_i (r'_i)^2} + \epsilon}$$

4. **Loss:**
   $$\mathcal{L}_{\text{spearman}} = 1 - \rho_s$$

**Alternative Backends:**
- **torchsort**: Uses fast-soft-sort (O(n log n)) with compiled kernels
- **diffsort**: Uses differentiable sorting networks (O(n²(log n)²))
- **rank-relax**: Optimized Rust implementation with analytical gradients

### 3.5 NeuralNDCG Loss

Approximates Normalized Discounted Cumulative Gain:

1. **Convert ICF to relevance:**
   $$r_i = 1 - y_i$$ (lower ICF = higher relevance)

2. **Compute DCG:**
   $$\text{DCG}@k = \sum_{i=1}^k \frac{r_{\pi(i)}}{\log_2(i + 1)}$$
   where $\pi$ is the ranking induced by $\hat{y}$ (descending)

3. **Ideal DCG:**
   $$\text{IDCG}@k = \sum_{i=1}^k \frac{r_{\pi^*(i)}}{\log_2(i + 1)}$$
   where $\pi^*$ is the ideal ranking (by $r$ descending)

4. **Loss:**
   $$\mathcal{L}_{\text{ndcg}} = 1 - \frac{\text{DCG}@k}{\text{IDCG}@k}$$

### 3.6 Listwise Ranking Losses

#### 3.6.1 LambdaRank Loss

Gradient-based ranking loss:

$$\lambda_{ij} = |\Delta\text{NDCG}_{ij}| \cdot \sigma(-\sigma_p(\hat{y}_i - \hat{y}_j))$$

$$\mathcal{L}_{\text{lambdarank}} = -\frac{1}{B(B-1)}\sum_{i \neq j} \lambda_{ij} \cdot (\hat{y}_i - \hat{y}_j)$$

where:
- $\Delta\text{NDCG}_{ij}$ = change in NDCG if $i$ and $j$ are swapped
- Approximated by $|r_i - r_j|$ where $r_i = 1 - y_i$
- $\sigma_p$ = smoothing parameter (default: 1.0)

#### 3.6.2 Approximate NDCG Loss

Differentiable approximation using softmax:

1. **Soft ranking weights:**
   $$w_i = \text{softmax}(\hat{y}_i / T)$$
   where $T$ = temperature (default: 1.0)

2. **Approximate DCG:**
   $$\text{DCG}_{\text{approx}} = \sum_i w_{\pi(i)} \cdot \frac{2^{y_{\pi(i)}} - 1}{\log_2(i + 1)}$$

3. **Loss:**
   $$\mathcal{L}_{\text{approx-ndcg}} = 1 - \frac{\text{DCG}_{\text{approx}}}{\text{IDCG}}$$

## 4. Evaluation Metrics

### 4.1 Regression Metrics

**Mean Absolute Error:**
$$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |\hat{y}_i - y_i|$$

**Root Mean Squared Error:**
$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2}$$

**Median Absolute Error:**
$$\text{MedAE} = \text{median}(|\hat{y}_i - y_i|)$$

### 4.2 Correlation Metrics

**Spearman Rank Correlation:**
$$\rho_s = \frac{\sum_i (r_{\hat{y},i} - \bar{r}_{\hat{y}})(r_{y,i} - \bar{r}_y)}{\sqrt{\sum_i (r_{\hat{y},i} - \bar{r}_{\hat{y}})^2}\sqrt{\sum_i (r_{y,i} - \bar{r}_y)^2}}$$

where $r_{\hat{y},i}$ and $r_{y,i}$ are ranks of $\hat{y}_i$ and $y_i$ respectively.

**Pearson Correlation:**
$$\rho_p = \frac{\sum_i (\hat{y}_i - \bar{\hat{y}})(y_i - \bar{y})}{\sqrt{\sum_i (\hat{y}_i - \bar{\hat{y}})^2}\sqrt{\sum_i (y_i - \bar{y})^2}}$$

**Kendall's Tau:**
$$\tau = \frac{2}{N(N-1)}\sum_{i<j} \text{sign}(\hat{y}_i - \hat{y}_j) \cdot \text{sign}(y_i - y_j)$$

### 4.3 Ranking Metrics

**NDCG@k:**
$$\text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}$$

where relevance $r_i = 1 - y_i$ (lower ICF = higher relevance).

**Mean Average Precision (MAP):**
$$\text{MAP} = \frac{1}{|\mathcal{R}|}\sum_{r \in \mathcal{R}} \text{AP}(r)$$

where $\mathcal{R}$ is the set of relevant items and $\text{AP}(r)$ is average precision for item $r$.

**Mean Reciprocal Rank (MRR):**
$$\text{MRR} = \frac{1}{|\mathcal{R}|}\sum_{r \in \mathcal{R}} \frac{1}{\text{rank}(r)}$$

### 4.4 Calibration Metrics

**Expected Calibration Error (ECE):**
$$\text{ECE} = \sum_{b=1}^B \frac{n_b}{N} |\bar{\hat{y}}_b - \bar{y}_b|$$

where $B$ bins partition $[0,1]$, $n_b$ = samples in bin $b$, $\bar{\hat{y}}_b$ = mean prediction in bin $b$, $\bar{y}_b$ = mean target in bin $b$.

**Brier Score:**
$$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2$$

### 4.5 Confidence Intervals

Bootstrap confidence intervals (default: 1000 samples, 95% confidence):

For metric $M$:
1. Sample with replacement: $\{(y_i, \hat{y}_i)\}_{i=1}^N \to \{(y_j, \hat{y}_j)\}_{j=1}^N$ (bootstrap sample)
2. Compute $M_b$ on bootstrap sample
3. Repeat $B$ times (default: 1000)
4. CI: $[M_{(\alpha/2)}, M_{(1-\alpha/2)}]$ where $M_{(p)}$ is $p$-th percentile

## 5. Training Procedure

### 5.1 Optimization

**Optimizer:** AdamW with component-specific learning rates:
- Embedding: $lr_e = lr \cdot \alpha_e$ (default: $\alpha_e = 0.1$)
- Convolutional layers: $lr_c = lr \cdot \alpha_c$ (default: $\alpha_c = 1.0$)
- MLP head: $lr_h = lr \cdot \alpha_h$ (default: $\alpha_h = 1.0$)

**Learning Rate Schedules:**

1. **ReduceLROnPlateau:**
   $$lr_{t+1} = \begin{cases}
   lr_t \cdot \gamma & \text{if } \text{val\_spearman}_t \text{ not improved for } p \text{ epochs} \\
   lr_t & \text{otherwise}
   \end{cases}$$
   where $\gamma$ = factor (default: 0.5), $p$ = patience (default: 8-12)

2. **Cosine Annealing with Warmup:**
   $$lr_t = \begin{cases}
   lr_0 \cdot \frac{t}{T_w} & \text{if } t < T_w \\
   lr_0 \cdot \frac{1 + \cos(\pi \frac{t - T_w}{T - T_w})}{2} & \text{if } t \geq T_w
   \end{cases}$$
   where $T_w$ = warmup epochs (default: 5), $T$ = total epochs

### 5.2 Regularization

**Gradient Clipping:**
$$\text{clip}(\mathbf{g}, c) = \mathbf{g} \cdot \min\left(1, \frac{c}{\|\mathbf{g}\|}\right)$$
where $c$ = max norm (default: 1.0)

**Dropout:** Applied in MLP head with probability $p$ (default: 0.3-0.4)

**Weight Decay:** L2 regularization with coefficient $\lambda_{wd}$ (default: $10^{-4}$)

### 5.3 Early Stopping

Stop training if validation Spearman correlation doesn't improve for $p$ epochs (default: 8-12) with minimum improvement $\delta$ (default: 0.0001).

### 5.4 Pair Generation

For ranking loss, pairs are generated via weighted sampling:

1. **Valid pairs:** $\mathcal{P} = \{(i,j) : y_i < y_j, y_j - y_i \geq m_{\min}\}$
2. **Weights:** $w_{ij} = \frac{y_j - y_i}{\sum_{(k,l) \in \mathcal{P}} (y_l - y_k)}$
3. **Sample:** $n_p$ pairs from $\mathcal{P}$ with probabilities $w_{ij}$

This emphasizes pairs with larger ICF differences, providing stronger learning signal.

## 6. Model Initialization

**Embedding:** $\mathcal{N}(0, 0.1^2)$

**Convolutional/Linear layers:** Kaiming normal initialization:
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$
where $n_{\text{in}}$ = number of input features

**Final layer:**
- Weights: $W \gets 0.1 \cdot W$ (scaled down)
- Bias: $b \gets \bar{y}$ (mean ICF, default: 0.4)

This prevents initial saturation and starts near expected output range.

## 7. Data Representation

**Input Encoding:**
- Words encoded as UTF-8 byte sequences: $w \to [b_1, b_2, \ldots, b_L] \in \{0, \ldots, 255\}^L$
- Padded/truncated to fixed length $L$ (default: 20)
- Padding token: 0

**Output:**
- Normalized ICF: $y \in [0, 1]$
- Clamped predictions: $\hat{y} = \text{clamp}(f(x), 0, 1)$

## 8. Notation Summary

| Symbol | Description |
|--------|-------------|
| $w$ | Word |
| $c_w$ | Word frequency count |
| $T$ | Total tokens in corpus |
| $y_w$ | Normalized ICF score for word $w$ |
| $\hat{y}$ | Model prediction |
| $B$ | Batch size |
| $L$ | Maximum sequence length |
| $d_e$ | Embedding dimension |
| $d_c$ | Convolutional channels |
| $d_h$ | Hidden dimension |
| $\lambda_r, \lambda_s, \lambda_n, \lambda_l$ | Loss component weights |
| $\delta$ | Huber loss threshold |
| $m$ | Ranking margin |
| $\tau$ | Temperature/regularization strength |
| $\sigma$ | Sigmoid function |
| $\rho_s, \rho_p, \tau$ | Spearman, Pearson, Kendall correlations |
| $lr$ | Learning rate |
| $\gamma$ | Learning rate decay factor |
| $p$ | Patience (early stopping) |
| $\alpha_e, \alpha_c, \alpha_h$ | Component-specific LR multipliers |

