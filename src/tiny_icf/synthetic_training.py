"""Synthetic training augmentation for OOV calibration.

Goal:
- Teach the model to separate *plausible composed words* from *gibberish*.

Approach:
- Generate composed-but-plausible words via prefix/suffix composition of common stems.
- Generate length-matched gibberish strings.
- Assign pseudo-labels:
  - composed words: mapped from character-bigram surprisal percentile into the real ICF distribution
  - gibberish: 1.0 (very rare)

This is optional and controlled via flags in `tiny_icf.train`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from tiny_icf.synthetic_oov import choose_bases, generate_composed_words, generate_gibberish_words


@dataclass(frozen=True)
class SyntheticOOVConfig:
    n_composed: int = 0
    common_k: int = 10_000
    seed: int = 42
    max_len: int = 20
    eps: float = 1e-12
    # Pseudo-label shaping for composed-but-plausible tokens.
    #
    # Goal: composed words should be "rare-ish" but clearly separated from gibberish (=1.0).
    # We map bigram-surprisal percentiles into a *band* around `composed_icf_center`
    # instead of the extreme tail of the real ICF distribution.
    composed_icf_center: float = 0.75
    composed_icf_spread: float = 0.2  # yields roughly [0.65, 0.85] for pct in [0, 1]


def _build_bigram_freqs(word_counts: Dict[str, int]) -> Dict[str, float]:
    """Token-weighted character bigram frequencies."""

    bigram_counts = Counter()
    for word, count in word_counts.items():
        w = word.lower()
        for i in range(len(w) - 1):
            bigram_counts[w[i : i + 2]] += int(count)

    total = float(sum(bigram_counts.values()))
    if total <= 0:
        return {}
    return {bg: c / total for bg, c in bigram_counts.items()}


def _mean_bigram_surprisal(word: str, bigram_freqs: Dict[str, float], *, eps: float) -> float:
    w = word.lower()
    if len(w) < 2:
        return float("inf")
    freqs = [bigram_freqs.get(w[i : i + 2], eps) for i in range(len(w) - 1)]
    freqs = np.maximum(np.asarray(freqs, dtype=np.float64), eps)
    return float(np.mean(-np.log(freqs)))


def _percentiles_from_reference(scores: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """Return percentile in [0, 1] for each score, relative to ref_sorted."""

    if len(ref_sorted) == 0:
        return np.full_like(scores, 0.5, dtype=np.float64)
    idx = np.searchsorted(ref_sorted, scores, side="right")
    return idx.astype(np.float64) / float(len(ref_sorted))


def _quantile_from_sorted(sorted_values: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Quantile lookup using linear interpolation; q in [0, 1]."""

    if len(sorted_values) == 0:
        return np.full_like(q, 0.5, dtype=np.float64)
    if len(sorted_values) == 1:
        return np.full_like(q, float(sorted_values[0]), dtype=np.float64)

    q = np.clip(q, 0.0, 1.0)
    pos = q * float(len(sorted_values) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.ceil(pos).astype(int)
    w = pos - lo
    return (1.0 - w) * sorted_values[lo] + w * sorted_values[hi]


def generate_synthetic_oov_pairs(
    *,
    word_counts: Dict[str, int],
    word_icf: Dict[str, float],
    config: SyntheticOOVConfig,
) -> List[Tuple[str, float]]:
    """
    Generate (word, pseudo_icf) pairs for synthetic OOV augmentation.

    Returns a list containing composed pairs + gibberish pairs.
    """

    if config.n_composed <= 0:
        return []

    sorted_by_count = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)
    common_k = min(int(config.common_k), len(sorted_by_count))
    common_words = [w for w, _ in sorted_by_count[:common_k]]
    real_set = set(word_counts.keys())

    bases = choose_bases(
        common_words, seed=int(config.seed), max_bases=min(5000, len(common_words))
    )
    composed = generate_composed_words(
        bases=bases, real_set=real_set, n=int(config.n_composed), seed=int(config.seed)
    )
    gibberish = generate_gibberish_words(
        lengths=[len(w) for w in composed],
        real_set=real_set,
        n=len(composed),
        seed=int(config.seed) + 1,
    )

    # Reference distribution (simple tokens only) for percentile calibration.
    ref_words: List[str] = [
        w for w in word_counts.keys() if w.isascii() and w.islower() and w.isalpha()
    ]

    bigram_freqs = _build_bigram_freqs(word_counts)
    ref_scores = np.array(
        [_mean_bigram_surprisal(w, bigram_freqs, eps=float(config.eps)) for w in ref_words],
        dtype=np.float64,
    )
    ref_scores_sorted = np.sort(ref_scores)

    comp_scores = np.array(
        [_mean_bigram_surprisal(w, bigram_freqs, eps=float(config.eps)) for w in composed],
        dtype=np.float64,
    )
    comp_pct = _percentiles_from_reference(comp_scores, ref_scores_sorted)

    # Shape the pseudo-labels into a bounded band in ICF-space.
    # This avoids teaching the model that "plausible composition" should saturate to 1.0.
    center = float(config.composed_icf_center)
    spread = float(config.composed_icf_spread)
    comp_pseudo_icf = np.clip(center + spread * (comp_pct - 0.5), 0.0, 1.0)

    composed_pairs = [(w, float(y)) for w, y in zip(composed, comp_pseudo_icf)]
    gibberish_pairs = [(w, 1.0) for w in gibberish]

    return composed_pairs + gibberish_pairs
