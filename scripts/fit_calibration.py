#!/usr/bin/env -S uv run
"""Fit affine calibration (a, b) on a held-out set: min MSE(target, a + b * pred).

Learned from data; no hand-picked words. Usage:
  uv run python scripts/fit_calibration.py --model models/multitask_all_fronts_v3b.pt --data data/word_frequency.csv
  uv run python scripts/fit_calibration.py --model models/multitask.pt --data data/word_frequency.csv --cal-ratio 0.2 --output models/multitask.cal.json
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from tiny_icf.calibration import save_calibration
from tiny_icf.checkpoint import load_model
from tiny_icf.data import (
    load_frequency_list,
    compute_normalized_icf,
    stratified_sample,
)
from tiny_icf.eval import evaluate_on_dataset
from tiny_icf.data import WordICFDataset


def main() -> int:
    p = argparse.ArgumentParser(description="Fit affine calibration from validation data")
    p.add_argument("--model", type=Path, required=True, help="Path to trained model .pt")
    p.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    p.add_argument(
        "--cal-ratio",
        type=float,
        default=0.2,
        help="Fraction of data to use for calibration (rest unused); default 0.2",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path; default <model>.cal.json",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )
    else:
        device = torch.device(args.device)

    model, _ = load_model(args.model, device=device)
    model.eval()

    word_counts, total_tokens = load_frequency_list(args.data)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    pairs = stratified_sample(
        word_icf,
        word_counts=word_counts,
        use_token_frequency=True,
    )
    random.shuffle(pairs)
    n_cal = max(1, int(len(pairs) * args.cal_ratio))
    cal_pairs = pairs[:n_cal]
    dataset = WordICFDataset(cal_pairs, max_length=20)

    result = evaluate_on_dataset(
        model,
        dataset,
        device,
        max_samples=None,
        batch_size=64,
    )
    preds = np.asarray(result["predictions"], dtype=np.float64)
    targets = np.asarray(result["targets"], dtype=np.float64)

    # Least squares: target ≈ a + b * pred
    X = np.column_stack([np.ones_like(preds), preds])
    beta, *_ = np.linalg.lstsq(X, targets, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    out_path = args.output or args.model.with_suffix(args.model.suffix + ".cal.json")
    save_calibration(out_path, a, b)

    calibrated = np.clip(a + b * preds, 0.0, 1.0)
    mae_before = np.mean(np.abs(preds - targets))
    mae_after = np.mean(np.abs(calibrated - targets))
    print(f"Calibration fit on {n_cal} samples")
    print(f"  a = {a:.6f}, b = {b:.6f}")
    print(f"  MAE before: {mae_before:.4f}, after: {mae_after:.4f}")
    print(f"  Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
