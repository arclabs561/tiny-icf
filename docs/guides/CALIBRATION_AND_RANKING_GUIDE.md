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

**Implemented:** `scripts/fit_calibration.py` fits (a, b) on a fraction of data (default 20%), writes `<model>.pt.cal.json`. `tiny_icf.calibration`: `load_calibration`, `save_calibration`, `apply_affine`. Use `--calibration <path>` in `tiny_icf-predict` and `evaluate_model.py`.

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

Evidence: Blondel et al. arxiv 2002.08871; codebase: `stratified_sample`, `use_token_frequency`, `SpearmanLoss` backends in `loss_unified` and `loss.py`.
