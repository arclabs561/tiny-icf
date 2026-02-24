"""Tests for affine calibration: apply_affine, load_calibration, save_calibration."""

import json
from pathlib import Path

import pytest

from tiny_icf.calibration import apply_affine, load_calibration, save_calibration


def test_apply_affine_formula():
    """apply_affine(pred, a, b) == clip(a + b * pred, 0, 1)."""
    a, b = 0.1, 0.9
    assert apply_affine(0.0, a, b) == pytest.approx(0.1)
    assert apply_affine(1.0, a, b) == pytest.approx(1.0)
    assert apply_affine(0.5, a, b) == pytest.approx(0.1 + 0.9 * 0.5)


def test_apply_affine_clipping():
    """Output is clipped to [0, 1]."""
    assert apply_affine(2.0, 0.0, 1.0) == 1.0
    assert apply_affine(-1.0, 0.0, 1.0) == 0.0
    assert apply_affine(0.5, 1.0, 1.0) == 1.0
    assert apply_affine(0.5, -0.5, 0.5) == 0.0


def test_save_load_roundtrip(tmp_path: Path):
    """save_calibration then load_calibration returns same (a, b)."""
    cal_path = tmp_path / "model.pt.cal.json"
    save_calibration(cal_path, 0.05, 0.95)
    loaded = load_calibration(cal_path)
    assert loaded is not None
    assert loaded == (0.05, 0.95)
    data = json.loads(cal_path.read_text())
    assert data["a"] == 0.05 and data["b"] == 0.95


def test_load_calibration_missing_returns_none(tmp_path: Path):
    """load_calibration on missing file returns None."""
    assert load_calibration(tmp_path / "nonexistent.cal.json") is None


def test_load_calibration_invalid_returns_none(tmp_path: Path):
    """load_calibration on invalid JSON returns None."""
    bad = tmp_path / "bad.cal.json"
    bad.write_text("not json")
    assert load_calibration(bad) is None
    bad.write_text("{}")
    # empty dict: get("a", 0.0), get("b", 1.0) so (0.0, 1.0) is valid
    assert load_calibration(bad) == (0.0, 1.0)
