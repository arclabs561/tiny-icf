"""Tests for ICF target modes."""

import pytest

from tiny_icf.data import compute_normalized_icf


def test_compute_normalized_icf_rank_mode_basic():
    word_counts = {"a": 100, "b": 50, "c": 10}
    total_tokens = sum(word_counts.values())

    out = compute_normalized_icf(word_counts, total_tokens, mode="rank", min_count=1)

    assert out["a"] == pytest.approx(0.0)
    assert out["b"] == pytest.approx(0.5)
    assert out["c"] == pytest.approx(1.0)


def test_compute_normalized_icf_rank_mode_singleton():
    word_counts = {"only": 10}
    out = compute_normalized_icf(word_counts, total_tokens=10, mode="rank", min_count=5)
    assert out["only"] == pytest.approx(0.5)

    out2 = compute_normalized_icf(word_counts, total_tokens=10, mode="rank", min_count=50)
    assert out2["only"] == pytest.approx(1.0)


def test_compute_normalized_icf_invalid_mode_raises():
    with pytest.raises(ValueError):
        compute_normalized_icf({"a": 1}, total_tokens=1, mode="nope")
