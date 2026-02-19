"""OOV calibration helpers.

This module is intentionally lightweight and dependency-free.

Why it exists:
- Many tiny-icf checkpoints use `output_activation="clamp"` during training/inference.
- For in-range examples this is fine, but for OOV/pseudo-words the model can produce
  `raw_output > 1.0`, which then gets hard-clamped to exactly `1.0`.
- That *destroys ordering information* in the extreme tail and causes the model to
  saturate on "plausible-but-unseen" composed words (e.g. "unfriendliness").

We keep the default model semantics untouched, and provide an *optional* saturation
fix for inference-time use-cases (Jabberwocky, OOV heuristics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SaturationFixConfig:
    """Config for mapping high-end clamp saturation back into (0, 1).

    The mapping only triggers when the model output is saturated at ~1.0 *and*
    the raw pre-activation output is > 1.0. In that case, we map `raw_output`
    through a logistic curve:

        y = sigmoid((raw_output - center) / scale)

    Choosing `center≈1.0` and `scale≈0.25` works well for the common tiny-icf
    UniversalICF checkpoints, where plausible OOV words often land in raw_output
    ~1.1–1.5, while gibberish can be ~2+.
    """

    eps: float = 1e-6
    center: float = 1.0
    scale: float = 0.25
    # Optional: incorporate the model's internal "confidence" feature (if available).
    # Positive weight means higher confidence -> higher fixed score (rarer).
    confidence_weight: float = 0.0
    confidence_center: float = 0.5


DEFAULT_SATURATION_FIX = SaturationFixConfig()


def _sigmoid(x: float) -> float:
    # Numerically stable enough for our small x range.
    return 1.0 / (1.0 + math.exp(-x))


def apply_saturation_fix(
    *,
    icf_score: float,
    raw_output: float,
    confidence: float | None = None,
    config: SaturationFixConfig = DEFAULT_SATURATION_FIX,
) -> tuple[float, bool]:
    """
    Apply an inference-time fix for high-end clamp saturation.

    Returns:
        (fixed_score, applied)
    """
    # Only adjust the high saturated tail.
    if icf_score >= 1.0 - config.eps and raw_output > 1.0 + config.eps:
        z = (raw_output - config.center) / config.scale
        if confidence is not None and config.confidence_weight != 0.0:
            z = z + config.confidence_weight * (confidence - config.confidence_center)
        fixed = _sigmoid(z)
        # Defensive clamp to [0, 1] (should already be true).
        fixed = 0.0 if fixed < 0.0 else 1.0 if fixed > 1.0 else fixed
        return fixed, True

    return icf_score, False
