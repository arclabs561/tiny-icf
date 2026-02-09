"""
Downstream evaluation harness for tiny-icf.

This script reports task-shaped metrics that better match likely usage:
- Common-word detection (AUROC) for top-K most frequent words.
- Gibberish-vs-common discrimination (AUROC) on length-matched random strings.
- OOV-style generalization (Spearman/MAE) on the training script's held-out split.
- Jabberwocky Protocol pass-rate.

It intentionally avoids writing large per-word prediction dumps.
"""

from __future__ import annotations

import argparse
import random
import string
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

# Make `src/` importable when running from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import (  # noqa: E402
    WordICFDataset,
    compute_normalized_icf,
    load_frequency_list,
    stratified_sample,
)
from tiny_icf.eval import evaluate_jabberwocky, evaluate_on_dataset  # noqa: E402
from tiny_icf.model import UniversalICF  # noqa: E402


def _auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    AUROC via rank-sum formula (tie-aware).

    y_true: 0/1 labels (1 = positive class)
    scores: higher should indicate more positive
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have same length")

    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(scores, method="average")  # 1..N
    sum_ranks_pos = float(ranks[pos].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _predict_words(model: UniversalICF, words: list[str], *, device: torch.device) -> np.ndarray:
    """Batch-predict ICF for a list of words."""
    pairs = [(w, 0.5) for w in words]  # dummy targets
    ds = WordICFDataset(pairs, max_length=20, augment_prob=0.0)
    dl = DataLoader(ds, batch_size=256, shuffle=False)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in dl:
            y = model(xb.to(device)).detach().cpu().numpy().reshape(-1)
            preds.append(y)
    return np.concatenate(preds, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Downstream evaluation for tiny-icf models")
    p.add_argument("--model", type=Path, required=True, help="Path to a trained model state_dict")
    p.add_argument("--data", type=Path, required=True, help="Frequency CSV (word,count)")
    p.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    p.add_argument("--seed", type=int, default=42, help="Seed (matches training default)")
    p.add_argument(
        "--common-k",
        type=int,
        default=10000,
        help="How many most-common words to treat as 'common' for downstream tasks",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (metrics only; no per-word dumps).",
    )

    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    word_counts, total_tokens = load_frequency_list(args.data)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    pairs = list(word_icf.items())
    words = [w for w, _ in pairs]

    # OOV-style metrics: replicate training's stratified split, evaluate only held-out words.
    # Do this *before* any evaluation that might consume numpy RNG state.
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    split_idx = int(len(samples) * 0.8)
    val_samples = samples[split_idx:]

    # Model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Full-dataset metrics (in-sample; includes training words if you trained on this same file)
    full_ds = WordICFDataset(pairs, max_length=20, augment_prob=0.0)
    full_eval = evaluate_on_dataset(model, full_ds, device, max_samples=None, batch_size=256)
    full_metrics = full_eval["metrics"]

    val_ds = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    val_eval = evaluate_on_dataset(model, val_ds, device, max_samples=None, batch_size=256)
    val_metrics = val_eval["metrics"]

    # Common-word detection AUROC
    sorted_by_count = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)
    common_k = min(args.common_k, len(sorted_by_count))
    common_words = [w for w, _ in sorted_by_count[:common_k]]
    common_set = set(common_words)

    # Use model predictions on the corpus words (already computed).
    model_icf = np.asarray(full_eval["predictions"]).reshape(-1)
    y_common = np.array([1 if w in common_set else 0 for w in words], dtype=int)
    common_auc = _auc_from_scores(y_common, -model_icf)  # lower ICF => more common

    # Gibberish-vs-common AUROC (length-matched gibberish)
    lengths = [len(w) for w in common_words]
    letters = string.ascii_lowercase

    real_set = set(word_counts.keys())

    gibberish: list[str] = []
    while len(gibberish) < common_k:
        L = random.choice(lengths)
        w = "".join(random.choice(letters) for _ in range(L))
        if w not in real_set:
            gibberish.append(w)

    gib_scores = _predict_words(model, gibberish, device=device)
    common_scores = _predict_words(model, common_words, device=device)
    gb_labels = np.array([0] * common_k + [1] * common_k, dtype=int)
    gb_scores = np.concatenate([common_scores, gib_scores], axis=0)
    gibberish_vs_common_auc = _auc_from_scores(gb_labels, gb_scores)

    # Jabberwocky Protocol
    jab = evaluate_jabberwocky(model, device)

    out: dict[str, Any] = {
        "device": str(device),
        "data_words": int(len(words)),
        "common_k": int(common_k),
        "full": {
            "spearman": float(full_metrics.get("spearman_corr", float("nan"))),
            "mae": float(full_metrics.get("mae", float("nan"))),
            "rmse": float(full_metrics.get("rmse", float("nan"))),
        },
        "oov_val": {
            "spearman": float(val_metrics.get("spearman_corr", float("nan"))),
            "mae": float(val_metrics.get("mae", float("nan"))),
            "rmse": float(val_metrics.get("rmse", float("nan"))),
            "n_samples": int(len(val_samples)),
        },
        "downstream": {
            "common_auc": float(common_auc),
            "gibberish_vs_common_auc": float(gibberish_vs_common_auc),
        },
        "jabberwocky": {
            "pass_rate": float(jab.get("pass_rate", float("nan"))),
            "passed": int(jab.get("passed_count", 0)),
            "total": int(jab.get("total_count", 0)),
        },
    }

    print("\nDownstream metrics")
    print(f"  full spearman: {out['full']['spearman']:.4f}  mae: {out['full']['mae']:.4f}")
    print(f"  oov  spearman: {out['oov_val']['spearman']:.4f}  mae: {out['oov_val']['mae']:.4f}")
    print(f"  common AUROC (top-{common_k}): {out['downstream']['common_auc']:.4f}")
    print(f"  gibberish-vs-common AUROC:     {out['downstream']['gibberish_vs_common_auc']:.4f}")
    print(f"  jabberwocky pass-rate:         {out['jabberwocky']['pass_rate']:.1%}")

    if args.output:
        import json

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()

