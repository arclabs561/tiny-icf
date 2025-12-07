#set page(margin: 2cm)
#set text(size: 11pt)
#set heading(numbering: "1.")

// Code highlighting available via codly package if installed
// #import "@preview/codly:0.1.0": *
// #show: codly.with(theme: "github-dark")

#title[
  Theoretical Bounds for Multi-Task Output Features
]

#text(fill: gray)[
  ICF Estimation Project · #datetime.today().display()
]

#par(justify: true)[
  This document establishes theoretical bounds and expected ranges for multi-task learning outputs beyond ICF prediction: Language Detection, Era Classification, Temporal ICF Prediction, and Text Reduction.
]

== Notation

- #strong[Accuracy]: Classification accuracy
- #strong[L_CE]: Cross-entropy loss
- #strong[L_MSE]: Mean squared error loss
- #strong[ρ]: Spearman rank correlation coefficient
- #strong[Regret]: Embedding regret (1 - cosine similarity)
- #strong[W₂]: Wasserstein-2 distance (for optimal transport formulation)
- #strong[I(S; Embedding)]: Mutual information between selected words S and embedding
- #strong[H(Embedding)]: Shannon entropy of embedding distribution
- #strong[N]: Number of classes (languages, eras)
- #strong[k]: Number of words to keep (text reduction budget)

== Language Detection

=== Task Description

Predict the language of a word from character patterns. This is a multi-class classification task (typically 10+ languages).

=== Theoretical Bound

#align(center)[
  $ "Acc"_"language" = frac(1, |D|) sum_((x, y) in D) 1[hat(y)(x) = y] $
]

where `hat(y)(x)` is the predicted language and `y` is the true language.

=== Expected Range

#let lang_bounds = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Expected Range for Character-Level Models:]
  - #strong[Best case:] Accuracy ∈ [0.70, 0.85] (character patterns are language-specific)
  - #strong[Good:] Accuracy ∈ [0.60, 0.70]
  - #strong[Acceptable:] Accuracy ∈ [0.50, 0.60]
  - #strong[Poor:] Accuracy < 0.50 (worse than random for balanced classes)
]

#lang_bounds

=== Mathematical Foundation

- For `N` languages with balanced classes, random baseline: Accuracy_random = 1/N
- For `N = 10` languages: Accuracy_random = 0.10
- Character patterns (n-grams, character frequency) are language-specific
- Expected accuracy: Accuracy ∈ [0.60, 0.85] depending on language similarity
- Upper bound: Accuracy ≤ 1 - H(Language|Characters) / H(Language) (Fano's inequality)

== Era Classification

=== Task Description

Predict the historical era when a word was commonly used (e.g., 1800s, 1900s, 2000s). This is a multi-class classification task (typically 3-5 eras).

=== Expected Range

#let era_bounds = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Expected Range for Character-Level Models:]
  - #strong[Best case:] `0.50 - 0.70` accuracy (some temporal patterns exist)
  - #strong[Good:] `0.40 - 0.50` accuracy
  - #strong[Acceptable:] `0.30 - 0.40` accuracy
  - #strong[Poor:] `< 0.30` accuracy (worse than random for 3-5 classes)
]

#era_bounds

=== Mathematical Foundation

- For N eras with balanced classes, random baseline = 1/N
- For 5 eras: random = 0.20
- Character patterns change over time (spelling reforms, new words)
- But changes are subtle and may not be captured by character-level models
- Expected accuracy: `0.30 - 0.60` (lower than language detection)

== Temporal ICF Prediction

=== Task Description

Predict ICF scores across multiple decades (e.g., 1800, 1900, 2000). This is a regression task with temporal consistency constraints.

=== Expected Range

#let temporal_bounds = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Expected Range for Character-Level Models:]
  - #strong[Best case:] `0.15 - 0.18` Spearman per decade (similar to ICF)
  - #strong[Good:] `0.12 - 0.15` Spearman per decade
  - #strong[Acceptable:] `0.10 - 0.12` Spearman per decade
  - #strong[Poor:] `< 0.10` Spearman per decade
]

#temporal_bounds

=== Mathematical Foundation

- Each decade has its own ICF distribution
- Character patterns → ICF is still indirect (same bound as ICF)
- Temporal consistency helps: predictions should be smooth across decades
- Expected Spearman: `0.12 - 0.18` (similar to ICF, but may be slightly lower due to temporal complexity)

== Text Reduction (Embedding Regret Minimization)

=== Task Description

Minimize embedding regret when reducing text by selecting a subset of words that preserves the original embedding as much as possible. This is a #strong[ranking + embedding similarity] task that can be #strong[disjoint from ICF prediction] (doesn't require ICF scores, but can use them as a heuristic).

#strong[Key Insight:] The task is to find the minimal "path" of embedding regret - i.e., select words such that the embedding of the reduced text is as close as possible to the original embedding.

=== Theoretical Bound

#let reduction_bounds = block(
  fill: luma(240),
  radius: 4pt,
  inset: 8pt,
)[
  #strong[Bound:] Depends on embedding quality, word selection strategy, and whether ICF is used

  #strong[Expected Range for Character-Level Models:]
  - #strong[Best case:] `0.05 - 0.15` regret (cosine distance)
  - #strong[Good:] `0.15 - 0.30` regret
  - #strong[Acceptable:] `0.30 - 0.50` regret
  - #strong[Poor:] `> 0.50` regret
]

#reduction_bounds

=== Mathematical Foundation

#strong[Regret] = `1 - cosine_similarity(original_embedding, reduced_embedding)`

#strong[Path Regret:] Cumulative embedding change along the reduction path

- Perfect reduction (keeping all important words): regret ≈ 0.0
- Random reduction: regret ≈ 0.5 - 0.7
- #strong[ICF-based reduction:] regret ≈ 0.15 - 0.30 (if ICF scores are accurate)
- #strong[Direct embedding-based reduction] (disjoint from ICF): regret ≈ 0.10 - 0.25 (potentially better, as it directly optimizes embedding similarity)
- Expected regret: `0.15 - 0.30` for ICF-based, `0.10 - 0.25` for direct embedding-based

=== Connection to ICF

#strong[Option 1 (Coupled):] Text reduction uses ICF scores to rank words (rare words = important)
- Better ICF prediction → better text reduction
- Multi-task learning can improve both by learning shared features

#strong[Option 2 (Disjoint):] Text reduction directly optimizes embedding similarity without ICF
- Can be trained independently of ICF
- May perform better (direct optimization vs proxy via ICF)
- Still benefits from shared character-level features in multi-task setup

== Summary Table

#table(
  columns: 6,
  align: center,
  stroke: 0.5pt,
  [*Task*],
  [*Metric*],
  [*Best Case*],
  [*Good*],
  [*Acceptable*],
  [*Poor*],
  [Language Detection], [Accuracy], [0.70-0.85], [0.60-0.70], [0.50-0.60], [< 0.50],
  [Language Detection], [Classification Loss], [0.2-0.5], [0.5-0.7], [0.7-1.0], [> 1.0],
  [Era Classification], [Accuracy], [0.50-0.70], [0.40-0.50], [0.30-0.40], [< 0.30],
  [Era Classification], [Classification Loss], [0.5-1.0], [1.0-1.5], [1.5-2.0], [> 2.0],
  [Temporal ICF], [Spearman], [0.15-0.18], [0.12-0.15], [0.10-0.12], [< 0.10],
  [Temporal ICF], [Consistency Loss], [0.0-0.05], [0.05-0.10], [0.10-0.20], [> 0.20],
  [Text Reduction], [Regret], [0.05-0.15], [0.15-0.30], [0.30-0.50], [> 0.50],
  [Text Reduction], [Path Regret], [0.10-0.20], [0.20-0.40], [0.40-0.60], [> 0.60],
  [Text Reduction], [Ranking Loss], [0.0-0.1], [0.1-0.2], [0.2-0.3], [> 0.3],
)

== Practical Guidelines

=== Multi-Task Balance

Monitor task weight ratios:
- ICF should dominate (primary task)
- Auxiliary tasks should help, not hurt
- If auxiliary task accuracy is poor, reduce its weight

=== Convergence Indicators

- #strong[Language accuracy:] Should increase to 0.60-0.70
- #strong[Era accuracy:] Should increase to 0.40-0.50
- #strong[Temporal Spearman:] Should increase to 0.12-0.15
- #strong[Text reduction regret:] Should decrease to 0.15-0.30

=== Warning Signs

- #strong[Language accuracy < 0.50:] Model not learning language features
- #strong[Era accuracy < 0.30:] Model not learning era features
- #strong[Temporal Spearman < 0.10:] Temporal prediction failing
- #strong[Text reduction regret > 0.50:] Reduction not preserving meaning
- #strong[One task dominates:] AMOO not balancing tasks

