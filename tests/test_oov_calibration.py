"""Tests for OOV calibration helpers."""

import math

from tiny_icf.oov_calibration import (
    DEFAULT_SATURATION_FIX,
    SaturationFixConfig,
    apply_saturation_fix,
)


def test_apply_saturation_fix_noop_when_not_saturated():
    fixed, applied = apply_saturation_fix(icf_score=0.73, raw_output=0.73)
    assert applied is False
    assert fixed == 0.73


def test_apply_saturation_fix_high_saturation_adjusts():
    # For saturated outputs, we should map raw_output back into (0, 1).
    fixed, applied = apply_saturation_fix(icf_score=1.0, raw_output=1.2)
    assert applied is True
    assert 0.0 < fixed < 1.0

    # Check against the configured logistic mapping.
    z = (1.2 - DEFAULT_SATURATION_FIX.center) / DEFAULT_SATURATION_FIX.scale
    expected = 1.0 / (1.0 + math.exp(-z))
    assert abs(fixed - expected) < 1e-12


def test_apply_saturation_fix_monotonic_in_raw_output():
    # Larger raw_output should map to a larger fixed score (when applied).
    a, a_applied = apply_saturation_fix(icf_score=1.0, raw_output=1.2)
    b, b_applied = apply_saturation_fix(icf_score=1.0, raw_output=1.4)
    assert a_applied and b_applied
    assert a < b


def test_apply_saturation_fix_confidence_weight_affects_when_enabled():
    cfg = SaturationFixConfig(
        eps=DEFAULT_SATURATION_FIX.eps,
        center=DEFAULT_SATURATION_FIX.center,
        scale=DEFAULT_SATURATION_FIX.scale,
        confidence_weight=10.0,
        confidence_center=0.5,
    )
    low, low_applied = apply_saturation_fix(
        icf_score=1.0, raw_output=1.2, confidence=0.40, config=cfg
    )
    high, high_applied = apply_saturation_fix(
        icf_score=1.0, raw_output=1.2, confidence=0.60, config=cfg
    )
    assert low_applied and high_applied
    assert low < high
