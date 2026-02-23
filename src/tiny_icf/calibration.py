"""Learned affine calibration for ICF predictions (minimal heuristics).

Fit (a, b) on validation data to minimize MSE between a + b * pred and target;
apply at inference so outputs are better calibrated. No hand-picked anchor words.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple


def apply_affine(pred: float, a: float, b: float) -> float:
    """Apply affine calibration and clip to [0, 1]."""
    out = a + b * pred
    return max(0.0, min(1.0, float(out)))


def load_calibration(path: Path) -> Optional[Tuple[float, float]]:
    """Load (a, b) from a JSON file. Returns None if file missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        a = float(data.get("a", 0.0))
        b = float(data.get("b", 1.0))
        return (a, b)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def save_calibration(path: Path, a: float, b: float) -> None:
    """Save (a, b) to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"a": a, "b": b}, f, indent=2)


def calibration_path_for_model(model_path: Path) -> Path:
    """Default path for calibration file given a model path."""
    return model_path.with_suffix(model_path.suffix + ".cal.json")
