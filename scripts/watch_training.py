#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# ///
"""
Print a concise summary of the latest training run from a Lightning metrics.csv.

Usage:
    uv run python scripts/watch_training.py [metrics.csv path]
    uv run python scripts/watch_training.py  # auto-finds latest run
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def _find_latest_metrics() -> Path | None:
    root = Path("models")
    candidates = sorted(root.glob("**/metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def summarise(path: Path) -> None:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Group by epoch — keep rows that have val_loss (end-of-epoch summary rows)
    epochs: dict[int, dict] = {}
    for row in rows:
        epoch_str = row.get("epoch", "").strip()
        if not epoch_str:
            continue
        try:
            e = int(float(epoch_str))
        except ValueError:
            continue
        val_loss = row.get("val_loss", "").strip()
        if val_loss:
            epochs[e] = row

    if not epochs:
        print("No epoch-end rows found yet.")
        return

    header = f"{'Ep':>3}  {'train_loss':>10}  {'val_loss':>9}  {'spearman':>8}  {'mae':>7}  {'icf_loss':>8}  {'lang':>6}  {'era':>6}  {'hyg':>6}  {'temp':>6}"
    print(header)
    print("-" * len(header))
    for e in sorted(epochs.keys()):
        r = epochs[e]
        tl   = r.get("train_loss_epoch", "").strip() or r.get("train_loss_step", "").strip()
        vl   = r.get("val_loss", "").strip()
        sp   = r.get("val_spearman_corr", "").strip()
        mae  = r.get("val_mae", "").strip()
        icf  = r.get("val_loss_icf", "").strip()
        lang = r.get("val_loss_language", "").strip()
        era  = r.get("val_loss_era", "").strip()
        hyg  = r.get("val_loss_hygiene", "").strip()
        tmp  = r.get("val_loss_temporal", "").strip()

        def _f(v: str, fmt: str = ".4f") -> str:
            try:
                return format(float(v), fmt)
            except (ValueError, TypeError):
                return "-"

        print(
            f"{e:>3}  {_f(tl, '.3f'):>10}  {_f(vl, '.4f'):>9}  {_f(sp, '.4f'):>8}"
            f"  {_f(mae, '.4f'):>7}  {_f(icf, '.4f'):>8}  {_f(lang, '.3f'):>6}"
            f"  {_f(era, '.3f'):>6}  {_f(hyg, '.3f'):>6}  {_f(tmp, '.3f'):>6}"
        )

    latest_e = max(epochs.keys())
    r = epochs[latest_e]
    lr = r.get("lr-AdamW", r.get("learning_rate", "")).strip()
    print()
    print(f"Latest: epoch {latest_e}, LR {lr}")
    print(f"Metrics file: {path}")


def main() -> None:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
    else:
        p = _find_latest_metrics()

    if p is None or not p.exists():
        print("No metrics.csv found. Pass a path as an argument.")
        return
    summarise(p)


if __name__ == "__main__":
    main()
