# Calibration and ranking guide

Research-backed improvements for ICF calibration and ranking quality, with minimal heuristics (no hand-picked anchor words or ad-hoc sample weights).

---

## 1. Sample from the data distribution

**Problem:** The head of the distribution ("the", "and") is under-represented in training. Stratified sampling uses **uniform within each stratum**, so "the" appears no more often than other head words.

**Fix:** Sample **weighted by token frequency** within strata, with **replacement** so high-count words can appear many times per epoch. The model then sees head words in proportion to the data; gradients come from the real distribution.

**Where:** `data.stratified_sample` with `use_token_frequency=True` (and `replace=True`); `lightning_data_multi_task.py` passes `word_counts` and `use_token_frequency=True`. No hand-picked word list.

---

## 2. Optimize Spearman directly (differentiable soft ranking)

**Problem:** Spearman is reported as a metric but the loss uses proxies (pairwise or sigmoid-based), so we do not directly optimize what we report.

**Fix:** **Differentiable Spearman** via soft sorting: Blondel et al., "Fast Differentiable Sorting and Ranking", ICML 2020 ([arxiv 2002.08871](https://arxiv.org/abs/2002.08871)). Loss \( \frac{1}{2}\|r - r_\Psi(\theta)\|^2 \) with soft ranks \( r_\Psi \). Implementations: **torchsort** (O(n log n)), **diffsort** (O(n²(log n)²)).

**Implemented:** `loss_unified.spearman_loss_tensor` with `spearman_method="auto"`: torchsort if available, else diffsort (default dependency), else built-in soft_rank. Training logs `Spearman loss backend: <torchsort|diffsort|built-in>`. CLI: `--spearman-reg-strength`, `--spearman-method auto|torchsort|diffsort|sigmoid`. See `docs/SPEARMAN_LOSS_BACKENDS.md` for backend details.

---

## 3. Calibration learned from validation

**Problem:** Common words (e.g. "the") can be over-predicted (too rare). Anchor-word fixes are heuristic.

**Fix:** **Affine calibration** on a held-out set: fit \( \hat{y} = a + b \cdot \text{pred} \) to minimize MSE; apply at inference.

**Implemented:** `scripts/fit_calibration.py` fits (a, b) on a fraction of data (default 20%), writes `<model>.pt.cal.json`. Affine regression calibration on held-out ICF targets; minimizes MSE. Improves MAE and Jabberwocky; Spearman may stay similar. `tiny_icf.calibration`: `load_calibration`, `save_calibration`, `apply_affine`. Use `--calibration <path>` in `tiny_icf-predict` and `evaluate_model.py`.

**Usage:** `just fit-calibration MODEL=models/<name>.pt DATA=data/word_frequency.csv` then `just eval-en` / `just eval-en-spearman` or `evaluate_model.py --calibration <name>.pt.cal.json`.

---

## 4. Optional: listwise ranking in the loss

**Problem:** Pairwise ranking does not enforce global order. Listwise losses (e.g. soft-rank MSE over a batch) align better with Spearman.

**Fix:** Use listwise options (`use_listwise_ranking` in the criterion); ensure batch size is large enough so that within-batch order is meaningful. Complements (2); no extra heuristics.

---

## 5. What to avoid

- **Anchor-word losses:** Prefer (1) + (3) over a fixed list of words and target ICF.
- **Hand-designed sample weights** (e.g. \( 1/\sqrt{\text{count}} \)): Prefer (1) — sample by actual frequency.
- **Synthetic OOV with a single hand-set target (e.g. 0.95):** Use a discriminative objective or derive targets from data; avoid one magic number for all OOV.

---

## Implementation order

1. **Frequency-weighted sampling** (1): datamodule change; no new deps; immediate effect on head calibration.
2. **Spearman backend** (2): ensure backend is torchsort or diffsort and `spearman_weight` is non-trivial; tune regularization if needed.
3. **Learned calibration** (3): fit (a, b) on val; apply at inference via `--calibration`.
4. **Listwise / larger batches** (4): if (1)–(3) are insufficient, enable listwise and/or increase batch size.

---

## Research (deeper): tiny-icf

### Differentiable Spearman and soft ranking

- **Blondel et al., "Fast Differentiable Sorting and Ranking," ICML 2020** ([arXiv 2002.08871](https://arxiv.org/abs/2002.08871)). Core reference for our Spearman loss: soft ranks via differentiable sorting; loss \( \frac{1}{2}\|r - r_\Psi(\theta)\|^2 \). O(n log n) with fast-soft-sort (torchsort); stable gradients.
- **Cuturi et al., "Differentiable Ranks and Sorting using Optimal Transport," arXiv 1905.11885.** Alternative view: sorting as optimal transport; Sinkhorn-based. Useful if you explore OT-based ranking.
- **Petersen et al., "Differentiable Sorting Networks" (arXiv 2105.04019, 2203.09630).** Relaxed sorting networks (diffsort-style); O(n²(log n)²). Our diffsort backend follows this line; good when torchsort is unavailable.
- **PiRank (Swezey et al., arXiv 2012.06731):** Differentiable sorting for learning-to-rank; ties ranking metrics to surrogate losses. Reinforces that optimizing soft-rank MSE aligns with Spearman.
- **Action:** Keep Blondel as the primary citation in the guide; document in code or SPEARMAN_LOSS_BACKENDS.md that diffsort = Petersen-style sorting networks, torchsort = Blondel fast-soft-sort.

### Calibration for regression (ICF)

- **Affine post-hoc:** Our approach (fit \( \hat{y} = a + b \cdot \text{pred} \) on held-out, minimize MSE) is standard for regression calibration. No single canonical citation; same idea as temperature scaling for classification but with two parameters.
- **Distribution calibration:** Song et al., "Distribution Calibration for Regression," arXiv 1905.06023 — calibrate full predictive distribution, not just point estimate. Overkill for ICF unless you need uncertainty bands.
- **Non-parametric regression calibration:** Song et al., "Non-Parametric Calibration of Probabilistic Regression," arXiv 1806.07690 — isotonic-style calibration for regression. Alternative if affine is too rigid; more data-hungry.
- **Action:** Document in `fit_calibration.py` or the guide that we use "affine (linear) regression calibration on held-out ICF targets; minimize MSE." If head words remain miscalibrated, consider isotonic (1806.07690) as an option.

### Word frequency and head/tail distribution

- **Unigram and rank–frequency:** Nikkarinen et al., "Modeling the Unigram Distribution," arXiv 2106.02289 — word frequency in corpora; sample frequency vs smoothed estimates. Ding, "A Crucial Parameter for Rank-Frequency Relation in Natural Languages," arXiv 2402.00271 — \( f \propto r^{-\alpha}(r+\gamma)^{-\beta} \) for rank–frequency; \( \gamma \) captures vocabulary growth. Supports that head (frequent) words need enough mass in training.
- **Rare words in NLMs:** Many papers (e.g. "Enriching Rare Word Representations," arXiv 1904.03799) address rare-word underfitting; our fix is the dual — **frequent-word underfitting** — addressed by sampling proportional to frequency (section 1).
- **Action:** When explaining frequency-weighted sampling, you can cite that rank–frequency in language is heavy-tailed and that uniform stratum sampling under-samples the head; our fix is to sample by count within strata.

### Evaluation (Jabberwocky, MAE, Spearman)

- **Jabberwocky protocol:** In-repo probe set (common words, rare words, gibberish, OOV). No direct citation; similar in spirit to minimal evaluation sets for robustness (e.g. checklisting). Document the bands and rationale in `evaluate_model.py` or a short eval doc.
- **Spearman vs MAE:** Spearman measures rank correlation (order); MAE measures point accuracy. Both matter for ICF: order for filtering/ranking, MAE for calibration. Our loss combines both (MAE/Huber + Spearman weight). PiRank (2012.06731) and LTR literature support optimizing a ranking surrogate directly.
- **Action:** In DATA_AND_MODELS or the guide, state that evaluation reports Jabberwocky (probe bands), MAE (calibration), and Spearman (ranking); calibration fit improves MAE/Jabberwocky, not necessarily Spearman.

---

Evidence: Blondel et al. arxiv 2002.08871; codebase: `stratified_sample`, `use_token_frequency`, `SpearmanLoss` backends in `loss_unified` and `loss.py`. Regression calibration: Song et al. 1905.06023, 1806.07690. Unigram/rank–frequency: 2106.02289, 2402.00271.
