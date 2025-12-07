#set page(margin: 2cm)
#set text(size: 11pt)
#set heading(numbering: "1.")

#import "@preview/codly:0.1.0": *

#show: codly.with(theme: "github-dark")

#title[
  Theoretical Bounds for Loss Components
]

#text(fill: gray)[
  ICF Estimation Project · #datetime.today().display()
]

#par(justify: true)[
  This document establishes theoretical bounds and expected ranges for each component of `ResearchAlignedICFLoss`. These bounds help us understand whether we're optimizing effectively, if loss components are in reasonable ranges, and what "good" performance looks like for each component.
]

== Notation

- #strong[L_huber]: Huber loss component
- #strong[L_rank]: Pairwise ranking loss component
- #strong[L_spearman]: Spearman correlation loss (1 - ρ)
- #strong[L_focal]: Focal loss component
- #strong[L_asym]: Asymmetric penalty component
- #strong[L_mono]: Monotonicity loss component
- #strong[L_quantile]: Quantile regression loss component
- #strong[ρ]: Spearman rank correlation coefficient
- #strong[H(ICF)]: Shannon entropy of ICF distribution
- #strong[I(Characters; ICF)]: Mutual information between character patterns and ICF

== Spearman Loss

=== Theoretical Bound

#let spearman_bound = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Bound:] `1.0 - ρ_max`, where `ρ_max ≈ 0.18-0.19` for character-level models

  #strong[Expected Range:]
  - #strong[Best case:] `1.0 - 0.19 = 0.81`
  - #strong[Current best:] `1.0 - 0.1891 = 0.8109`
  - #strong[Typical:] `0.82 - 0.85`
]

#spearman_bound

=== Interpretation

- Lower is better (closer to 0.81 = better)
- Values > 0.85 suggest poor ranking
- Values < 0.81 are theoretically impossible for character-level models

=== Information-Theoretic Foundation

The maximum Spearman correlation is bounded by:

#align(center)[
  $ rho_max <= sqrt(frac(I(X; Y), H(Y))) approx 0.18-0.19 $
]

where $X$ represents character features and $Y$ represents ICF values.

This follows from the data processing inequality and the relationship between correlation and mutual information.

== Huber Loss

=== Theoretical Bound

#let huber_bound = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Bound:] Depends on data distribution and model capacity

  #strong[Expected Range for ICF:]
  - #strong[Best case:] `0.01 - 0.05` (very small errors)
  - #strong[Good:] `0.05 - 0.10` (small errors)
  - #strong[Acceptable:] `0.10 - 0.20` (moderate errors)
  - #strong[Poor:] `> 0.20` (large errors)
]

#huber_bound

=== Mathematical Foundation

Huber loss with `δ = 0.1`:

#align(center)[
  $ L_delta(a) = {
    0.5 a^2, "when" "abs"(a) <= 0.1; \
    0.1 ("abs"(a) - 0.05), "otherwise"
  } $
]

Where the first case applies when $"abs"(a) <= 0.1$ and the second case applies otherwise.

For ICF values in `[0, 1]`, typical errors are `0.1-0.3`. Expected loss: $E[L_delta] approx 0.05 - 0.15$ for well-trained models.

=== Interpretation

- Lower is better
- Values < 0.05 indicate excellent regression accuracy
- Values > 0.20 suggest model is not learning well

== Ranking Loss (Pairwise)

=== Theoretical Bound

#let ranking_bound = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Bound:] Depends on ranking method and data distribution

  #strong[Expected Range for ICF:]
  - #strong[Best case:] `0.0 - 0.05` (perfect pairwise ordering)
  - #strong[Good:] `0.05 - 0.15` (mostly correct ordering)
  - #strong[Acceptable:] `0.15 - 0.30` (some ordering errors)
  - #strong[Poor:] `> 0.30` (many ordering violations)
]

#ranking_bound

=== Mathematical Foundation

Margin-based ranking loss:

#align(center)[
  $ L_"rank" = "max"(0, "margin" - "pred_diff" * "sign"("target_diff")) $
]

With `margin = 0.1`, perfect ordering gives loss `= 0.0`. Random ordering gives loss `≈ 0.1` (margin value).

=== Theoretical Limitation

Research shows: "All convex per-edge surrogates are inconsistent for strict ranking". This means even perfect optimization may not achieve `loss = 0.0`. Expected minimum: `0.02 - 0.05` due to theoretical inconsistency.

== Focal Loss Component

=== Theoretical Bound

#let focal_bound = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Bound:] Depends on base loss and gamma parameter

  #strong[Expected Range for ICF:]
  - #strong[Best case:] `0.01 - 0.05` (focused on hard examples, most are easy)
  - #strong[Good:] `0.05 - 0.15` (balanced focus)
  - #strong[Acceptable:] `0.15 - 0.30` (many hard examples)
  - #strong[Poor:] `> 0.30` (too many hard examples, model struggling)
]

#focal_bound

=== Mathematical Foundation

Focal loss:

#align(center)[
  $ "FL" = (1 + "error")^"gamma" * "base_loss" $
]

With `γ = 2.0`, easy examples (error < 0.1) get down-weighted. Hard examples (error > 0.3) get up-weighted exponentially.

=== Expected Behavior

- Focal loss should be #strong[lower] than base loss (Huber)
- If focal loss > Huber loss, focal weighting is not helping
- Typical ratio: `focal_loss / huber_loss ≈ 0.5 - 0.8` (focal down-weights easy examples)

== Asymmetric Penalty

=== Theoretical Bound

#let asym_bound = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Bound:] Depends on asymmetry factor and error distribution

  #strong[Expected Range for ICF:]
  - #strong[Best case:] `0.0 - 0.02` (minimal asymmetric errors)
  - #strong[Good:] `0.02 - 0.05` (some asymmetric errors, but small)
  - #strong[Acceptable:] `0.05 - 0.10` (moderate asymmetric errors)
  - #strong[Poor:] `> 0.10` (many large asymmetric errors)
]

#asym_bound

=== Mathematical Foundation

Asymmetric penalty:

#align(center)[
  $ L_"asym" = {
    "factor" * "relu"("error"), "when common→rare"; \
    "relu"(-"error"), "when rare→common"
  } $
]

With `factor = 2.0`, common→rare errors are penalized 2× more.

== Monotonicity Loss

=== Theoretical Bound

#let mono_bound = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Bound:] Depends on feature-prediction correlation

  #strong[Expected Range for ICF:]
  - #strong[Best case:] `0.0` (perfect monotonicity)
  - #strong[Good:] `0.0 - 0.01` (minimal violations)
  - #strong[Acceptable:] `0.01 - 0.05` (some violations)
  - #strong[Poor:] `> 0.05` (many violations)
]

#mono_bound

=== Mathematical Foundation

Monotonicity loss:

#align(center)[
  $ L_"mono" = {
    "relu"(-"correlation"), "when increasing"; \
    "relu"("correlation"), "when decreasing"
  } $
]

For word length → ICF (increasing), correlation should be positive. Perfect monotonicity: correlation `= ±1.0`, loss `= 0.0`.

== Quantile Regression Loss

=== Theoretical Bound

#let quantile_bound = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Bound:] Depends on quantile and data distribution

  #strong[Expected Range for ICF:]
  - #strong[Best case:] `0.05 - 0.10` (good quantile prediction)
  - #strong[Good:] `0.10 - 0.20` (reasonable quantile prediction)
  - #strong[Acceptable:] `0.20 - 0.30` (moderate quantile errors)
  - #strong[Poor:] `> 0.30` (poor quantile prediction)
]

#quantile_bound

=== Mathematical Foundation

Quantile loss:

#align(center)[
  $ L_"quantile" = "max"("tau" * "error", ("tau" - 1) * "error") $
]

For `τ = 0.5` (median), loss `≈ 0.5 * MAE`. For `τ = 0.9` (90th percentile), loss is asymmetric (over-prediction penalized more).

== Adaptive Regularization Strength

=== Theoretical Bound

#let reg_bound = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Bound:] `1.0 / typical_difference`, where typical_difference is data scale

  #strong[Expected Range for ICF:]
  - #strong[Typical:] `5.0 - 20.0` (for ICF values in [0, 1])
  - #strong[High:] `20.0 - 50.0` (for very small differences)
  - #strong[Low:] `1.0 - 5.0` (for large differences)
]

#reg_bound

=== Mathematical Foundation

Adaptive regularization:

#align(center)[
  $ "reg_strength" = 1.0 / ("std" + "MAD") $
]

For ICF with `std ≈ 0.2-0.3`, `MAD ≈ 0.1-0.2`. Expected: `reg_strength ≈ 1.0 / 0.2 = 5.0`. Clamped to `[0.1, 100.0]` for stability.

== Summary Table

#table(
  columns: 6,
  align: center,
  stroke: 0.5pt,
  [*Component*],
  [*Best Case*],
  [*Good*],
  [*Acceptable*],
  [*Poor*],
  [*Interpretation*],
  [Spearman Loss], [0.81], [0.82-0.85], [0.85-0.90], [> 0.90], [Lower is better, bound: 0.81],
  [Huber Loss], [0.01-0.05], [0.05-0.10], [0.10-0.20], [> 0.20], [Lower is better],
  [Ranking Loss], [0.0-0.05], [0.05-0.15], [0.15-0.30], [> 0.30], [Lower is better, min: ~0.02],
  [Focal Loss], [0.01-0.05], [0.05-0.15], [0.15-0.30], [> 0.30], [Lower is better, < Huber],
  [Asymmetric Penalty], [0.0-0.02], [0.02-0.05], [0.05-0.10], [> 0.10], [Lower is better, < Huber],
  [Monotonicity Loss], [0.0], [0.0-0.01], [0.01-0.05], [> 0.05], [Lower is better, 0.0 = perfect],
  [Quantile Loss], [0.05-0.10], [0.10-0.20], [0.20-0.30], [> 0.30], [Lower is better, ≈ Huber],
  [Reg Strength], [5.0-20.0], [1.0-50.0], [-], [-], [Adaptive, not "better/worse"],
)

== Practical Guidelines

=== Component Ratios

Monitor these ratios to understand loss balance:

- #strong[Spearman / Total:] Should be dominant (0.7-0.9) since it's weighted 10×
- #strong[Ranking / Total:] Should be moderate (0.05-0.15) since it's weighted 0.5×
- #strong[Huber / Total:] Should be small (0.05-0.15) since it's base loss
- #strong[Focal / Huber:] Should be < 1.0 (focal down-weights easy examples)
- #strong[Asymmetric / Huber:] Should be < 0.5 (asymmetric is additional penalty)

=== Convergence Indicators

- #strong[Spearman loss:] Should decrease to ~0.81-0.85 and stabilize
- #strong[Huber loss:] Should decrease to ~0.05-0.10 and stabilize
- #strong[Ranking loss:] Should decrease to ~0.05-0.15 and stabilize
- #strong[Component ratios:] Should stabilize (not drift) during training

=== Warning Signs

- #strong[Spearman loss > 0.90:] Model is not learning ranking
- #strong[Huber loss > 0.20:] Model is not learning regression
- #strong[Ranking loss > 0.30:] Many pairwise ordering violations
- #strong[Focal loss > Huber loss:] Focal weighting not helping
- #strong[Asymmetric penalty > 0.10:] Systematic bias in predictions
- #strong[Monotonicity loss > 0.05:] Constraints violated or too strict

