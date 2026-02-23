"""Tests for stratified_sample (especially frequency-weighted with replacement)."""

import pytest
from collections import Counter

from tiny_icf.data import stratified_sample


def test_stratified_sample_frequency_weighted_draws_with_replacement():
    """With use_token_frequency=True, high-count words should appear multiple times."""
    word_icf = {f"w{i}": 0.1 + (i / 100) * 0.9 for i in range(100)}
    word_icf["the"] = 0.05
    word_icf["and"] = 0.08
    word_counts = {w: 1 for w in word_icf}
    word_counts["the"] = 100_000
    word_counts["and"] = 50_000

    out = stratified_sample(
        word_icf,
        word_counts=word_counts,
        use_token_frequency=True,
        max_samples=500,
    )
    c = Counter(w for w, _ in out)
    assert c.get("the", 0) >= 2, "the should appear multiple times with freq sampling"
    assert c.get("and", 0) >= 2, "and should appear multiple times with freq sampling"


def test_stratified_sample_uniform_at_most_once_per_word():
    """Without use_token_frequency, each word in a stratum appears at most once."""
    word_icf = {f"w{i}": 0.5 for i in range(20)}
    out = stratified_sample(word_icf, max_samples=100)
    c = Counter(w for w, _ in out)
    for w, count in c.items():
        assert count == 1, f"uniform sampling should not repeat {w}"
