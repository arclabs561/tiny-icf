# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""
Evaluate auxiliary "all-fronts" heads (hygiene / language / era / temporal).

This is intentionally lightweight: it uses heuristic labels (the same ones used for training)
and reports simple accuracy / MSE summaries.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tiny_icf.checkpoint import load_model
from tiny_icf.data import load_frequency_list
from tiny_icf.data_multi_task import ERA_CODES, HYGIENE_CODES, LANGUAGE_CODES, _label_hygiene
from tiny_icf.predict import word_to_bytes
from tiny_icf.temporal_detection import detect_era_patterns


def _load_temporal_csv(path: Path, decades: list[int]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get("word") or "").strip().lower()
            if not w:
                continue
            m: dict[int, float] = {}
            for dec in decades:
                col = f"icf_{int(dec)}"
                raw = row.get(col)
                if raw is None:
                    continue
                raw = str(raw).strip()
                if not raw:
                    continue
                try:
                    m[int(dec)] = float(raw)
                except ValueError:
                    continue
            if m:
                out[w] = m
    return out


def _batch_bytes(words: list[str], max_length: int) -> torch.Tensor:
    return torch.cat([word_to_bytes(w, max_length=max_length) for w in words], dim=0)


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate multi-task (all-fronts) heads")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="Frequency CSV (used to sample tokens)")
    p.add_argument("--n", type=int, default=2000, help="Number of tokens to sample")
    p.add_argument("--max-length", type=int, default=20)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--temporal-data", type=Path, default=None, help="Historical ICF CSV with icf_YYYY columns")
    p.add_argument("--temporal-decades", type=str, default="1800,1900,2000")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, _ckpt = load_model(args.model, device=device)
    model.eval()

    # Sample tokens: raw includes noise, clean excludes noise (per preprocessing).
    raw_counts, _ = load_frequency_list(args.data, filter_noise=False)
    clean_counts, _ = load_frequency_list(args.data, filter_noise=True)
    raw_tokens = list(raw_counts.keys())
    clean_set = set(clean_counts.keys())

    if not raw_tokens:
        raise RuntimeError("No tokens loaded from --data")

    # Mixed sample (gives hygiene head something non-trivial).
    n = int(min(args.n, len(raw_tokens)))
    sample = random.sample(raw_tokens, k=n)

    # True labels
    y_hyg_true = np.array([_label_hygiene(w) for w in sample], dtype=int)

    # Language labels only for explicit lang:word tokens.
    y_lang_true = np.full((n,), -1, dtype=int)
    for i, w in enumerate(sample):
        if ":" in w:
            maybe_lang, _rest = w.split(":", 1)
            if maybe_lang in LANGUAGE_CODES:
                y_lang_true[i] = LANGUAGE_CODES.index(maybe_lang)

    # Era labels from heuristic detector (weak, but matches training labels).
    y_era_true = np.full((n,), -1, dtype=int)
    for i, w in enumerate(sample):
        eras = detect_era_patterns(w)
        if eras:
            top_era = max(eras.items(), key=lambda kv: kv[1])[0]
            if top_era in ERA_CODES:
                y_era_true[i] = ERA_CODES.index(top_era)

    # Temporal targets (optional)
    decades = [int(x.strip()) for x in args.temporal_decades.split(",") if x.strip()]
    temporal_map: dict[str, dict[int, float]] = {}
    if args.temporal_data is not None and args.temporal_data.exists():
        temporal_map = _load_temporal_csv(args.temporal_data, decades)

    # Inference
    bs = 256
    y_hyg_pred: list[int] = []
    y_lang_pred: list[int] = []
    y_era_pred: list[int] = []
    temporal_preds: list[np.ndarray] = []
    temporal_tgts: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, n, bs):
            batch_words = sample[i : i + bs]
            xb = _batch_bytes(batch_words, max_length=args.max_length).to(device)

            try:
                out = model(xb, return_all=True)  # type: ignore[misc]
            except Exception:
                out = {"icf": model(xb)}

            if isinstance(out, dict):
                if "hygiene" in out:
                    hyg = out["hygiene"]
                    y_hyg_pred.extend(torch.argmax(hyg, dim=-1).detach().cpu().tolist())
                if "language" in out:
                    lang = out["language"]
                    y_lang_pred.extend(torch.argmax(lang, dim=-1).detach().cpu().tolist())
                if "era" in out:
                    era = out["era"]
                    y_era_pred.extend(torch.argmax(era, dim=-1).detach().cpu().tolist())
                if "temporal" in out and hasattr(model, "temporal_decades") and temporal_map:
                    decs = list(getattr(model, "temporal_decades"))
                    temp = out["temporal"].detach().cpu().numpy()
                    for w, row in zip(batch_words, temp):
                        base = w.split(":", 1)[1] if ":" in w else w
                        if base in temporal_map:
                            tgt = np.array([temporal_map[base].get(d, np.nan) for d in decs], dtype=float)
                            if np.isfinite(tgt).all() and tgt.shape == row.shape:
                                temporal_preds.append(row.astype(float))
                                temporal_tgts.append(tgt.astype(float))

    # Hygiene accuracy (only if model produced predictions)
    out_metrics: dict[str, Any] = {"n": n, "device": str(device)}

    if y_hyg_pred:
        y_hyg_pred_np = np.array(y_hyg_pred[:n], dtype=int)
        acc = float((y_hyg_pred_np == y_hyg_true).mean())
        out_metrics["hygiene_acc"] = acc
    else:
        out_metrics["hygiene_acc"] = None

    # Language accuracy (only where ground-truth label exists)
    if y_lang_pred and (y_lang_true >= 0).any():
        y_lang_pred_np = np.array(y_lang_pred[:n], dtype=int)
        mask = y_lang_true >= 0
        out_metrics["language_acc_on_prefixed"] = float((y_lang_pred_np[mask] == y_lang_true[mask]).mean())
        out_metrics["language_n_prefixed"] = int(mask.sum())
    else:
        out_metrics["language_acc_on_prefixed"] = None

    # Era accuracy (only where heuristic label exists)
    if y_era_pred and (y_era_true >= 0).any():
        y_era_pred_np = np.array(y_era_pred[:n], dtype=int)
        mask = y_era_true >= 0
        out_metrics["era_acc_on_heuristic"] = float((y_era_pred_np[mask] == y_era_true[mask]).mean())
        out_metrics["era_n_labeled"] = int(mask.sum())
    else:
        out_metrics["era_acc_on_heuristic"] = None

    # Temporal MSE
    if temporal_preds and temporal_tgts:
        p_mat = np.stack(temporal_preds, axis=0)
        t_mat = np.stack(temporal_tgts, axis=0)
        out_metrics["temporal_mse"] = float(((p_mat - t_mat) ** 2).mean())
        out_metrics["temporal_n"] = int(p_mat.shape[0])
    else:
        out_metrics["temporal_mse"] = None

    print(out_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

