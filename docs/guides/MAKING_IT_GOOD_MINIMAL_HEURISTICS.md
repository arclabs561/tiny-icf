# Making the model actually good (minimal heuristics)

Research-backed, low-heuristic improvements. No hand-picked anchor words or ad-hoc sample weights.

---

## 1. Sample from the data distribution (zero heuristics)

**Problem:** The head of the distribution ("the", "and") is under-represented in training. We use stratified sampling (head/body/tail) but **uniform within each stratum**, so "the" appears no more often than other head words.

**Fix:** Sample **weighted by token frequency** within strata. Then the model sees "the" as often as it appears in the world; gradients for the head come from real data, not a fixed list.

**Where:** `lightning_data_multi_task.py`. **Implemented:** we now pass `word_counts=train_word_counts` (and `val_word_counts`) and `use_token_frequency=True` so training and validation sample within each stratum by token frequency. No hand-picked word list; the data distribution drives the head.

---

## 2. Optimize Spearman directly (differentiable soft ranking)

**Problem:** Spearman is computed as a metric but the loss is a mix of MAE/Huber and a **proxy** for ranking (pairwise or sigmoid-based). So we don't directly optimize what we report.

**Fix:** Use **differentiable Spearman** via soft sorting (Blondel et al., "Fast Differentiable Sorting and Ranking", ICML 2020; [arxiv 2002.08871](https://arxiv.org/abs/2002.08871)). Loss = \( \frac{1}{2}\|r - r_\Psi(\theta)\|^2 \) where \( r_\Psi \) are soft ranks. Implementations: **torchsort** (O(n log n), recommended), **diffsort** (O(n²(log n)²)).

**Implemented:** `loss_unified.spearman_loss_tensor` prefers **torchsort** when `spearman_method` is `"auto"` (default) and torchsort is installed. Training uses `--spearman-method auto`; install with `uv sync --extra sorting` for O(n log n) differentiable Spearman. CLI: `--spearman-reg-strength 0.1`, `--spearman-method auto|torchsort|sigmoid`. Fallback is rank_relax or built-in sigmoid.

---

## 3. Calibration learned from validation (one scalar or affine map)

**Problem:** "the" is over-predicted (too rare). We could fix it with anchor words (heuristic) or with **learned** calibration.

**Fix:** **Affine calibration** for regression: on a held-out set, fit \( \hat{y} = a + b \cdot \text{pred} \) to minimize MSE. Two scalars learned from data; apply at inference.

**Implemented:**
- `scripts/fit_calibration.py` — fits (a, b) on a fraction of the data (default 20%), saves to `<model>.cal.json`.
- `tiny_icf.calibration` — `load_calibration`, `save_calibration`, `apply_affine`.
- `tiny_icf-predict --calibration <path>` and `evaluate_model.py --calibration <path>` apply the affine map to predictions.

Usage: `uv run python scripts/fit_calibration.py --model models/multitask_all_fronts_v3b.pt --data data/word_frequency.csv` then pass `--calibration models/multitask_all_fronts_v3b.pt.cal.json` to predict or evaluate_model.

---

## 4. Optional: listwise ranking in the loss

**Problem:** Pairwise ranking helps but doesn't enforce global order. Listwise losses (e.g. ListNet, or soft-rank MSE over a full batch) match the evaluation metric (Spearman) better.

**Fix:** We already have listwise options (`use_listwise_ranking` in the criterion). Ensure batches are large enough and the listwise term is enabled so that within each batch we optimize order over many words at once. This complements (2); no extra heuristics.

---

## 5. What to avoid

- **Anchor-word losses:** Fixing "the" with a hand list and target ICF is a heuristic. Prefer (1) + (3).
- **Hand-designed sample weights** (e.g. \( 1/\sqrt{\text{count}} \)): Prefer (1) — sample by actual frequency.
- **Synthetic OOV with hand-set target 0.95:** If you add OOV, either (a) use a separate discriminative objective (in-vocab vs OOV) or (b) derive targets from a held-out process; avoid a single magic number for "all OOV".

---

## Implementation order

1. **Frequency-weighted sampling** (1): one change in the datamodule; no new deps; immediate effect on head calibration.
2. **Verify Spearman backend** (2): ensure torchsort is used and spearman_weight is non-trivial; tune reg strength if needed.
3. **Learned calibration** (3): fit \( a, b \) on val; apply at inference. Small script or flag in predict.
4. **Listwise / larger batches** (4): if (1)–(3) aren’t enough, turn on listwise and/or increase batch size for ranking.

Evidence: Perplexity search (differentiable Spearman, temperature scaling); arxiv search (Blondel et al. 2002.08871); codebase grep (stratified_sample, use_token_frequency, SpearmanLoss backends).
