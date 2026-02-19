"""
Evaluate OOV calibration and clamp saturation behavior.

This script targets the main practical failure mode we've observed:
- models trained with output_activation="clamp" can produce raw_output > 1.0 for OOV/pseudo-words,
  which gets hard-clamped to 1.0 and destroys ordering information in the tail.

We generate two OOV sets:
- composed-but-plausible words (prefix/suffix composition of common stems)
- length-matched gibberish (random lowercase strings)

and report:
- saturation rate (fraction predicted exactly ~1.0)
- AUROC(gibberish vs composed) using ICF as the score
- before/after applying the optional saturation-fix mapping.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

# Make `src/` importable when running from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.checkpoint import load_model  # noqa: E402
from tiny_icf.data import load_frequency_list  # noqa: E402
from tiny_icf.oov_calibration import (  # noqa: E402
    DEFAULT_SATURATION_FIX,
    SaturationFixConfig,
    apply_saturation_fix,
)
from tiny_icf.predict import word_to_bytes  # noqa: E402
from tiny_icf.synthetic_oov import (
    choose_bases,
    generate_composed_words,
    generate_gibberish_words,
)  # noqa: E402


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


@dataclass(frozen=True)
class Predictions:
    model: np.ndarray
    raw: np.ndarray | None
    fixed: np.ndarray | None
    applied: np.ndarray | None
    confidence: np.ndarray | None


def _batch_predict(
    model: torch.nn.Module,
    words: list[str],
    *,
    device: torch.device,
    saturation_fix: bool,
    fix_config: SaturationFixConfig,
) -> Predictions:
    if not words:
        empty = np.zeros((0,), dtype=float)
        return Predictions(model=empty, raw=None, fixed=None, applied=None, confidence=None)

    xb = torch.cat([word_to_bytes(w, max_length=20) for w in words], dim=0).to(device)
    model.eval()

    with torch.no_grad():
        try:
            y, feats = model(xb, return_features=True)  # type: ignore[misc]
            y_np = y.detach().cpu().numpy().reshape(-1)
            raw = feats.get("raw_output", y).detach().cpu().numpy().reshape(-1)
            conf = feats.get("confidence", None)
            confidence = conf.detach().cpu().numpy().reshape(-1) if conf is not None else None
        except Exception:
            y = model(xb)
            y_np = y.detach().cpu().numpy().reshape(-1)
            raw = None
            confidence = None

    if not saturation_fix or raw is None:
        return Predictions(model=y_np, raw=raw, fixed=None, applied=None, confidence=confidence)

    fixed = np.zeros_like(y_np, dtype=float)
    applied = np.zeros_like(y_np, dtype=bool)
    for i, (score, ro) in enumerate(zip(y_np, raw)):
        conf_i = float(confidence[i]) if confidence is not None else None
        fx, ap = apply_saturation_fix(
            icf_score=float(score),
            raw_output=float(ro),
            confidence=conf_i,
            config=fix_config,
        )
        fixed[i] = fx
        applied[i] = ap
    return Predictions(model=y_np, raw=raw, fixed=fixed, applied=applied, confidence=confidence)


def _summarize_scores(name: str, scores: np.ndarray) -> dict[str, float]:
    if len(scores) == 0:
        return {"n": 0.0}
    return {
        "n": float(len(scores)),
        "mean": float(np.mean(scores)),
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "sat_1p0": float(np.mean(scores >= 1.0 - 1e-6)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate OOV calibration and clamp saturation")
    p.add_argument("--model", type=Path, required=True, help="Path to a trained model checkpoint")
    p.add_argument("--data", type=Path, required=True, help="Frequency CSV (word,count)")
    p.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n", type=int, default=2000, help="How many composed + gibberish words to generate"
    )
    p.add_argument(
        "--common-k",
        type=int,
        default=10000,
        help="How many most-common words to draw stems from (after filtering to simple tokens)",
    )
    p.add_argument(
        "--fix-center",
        type=float,
        default=float(DEFAULT_SATURATION_FIX.center),
        help="Saturation-fix center parameter (raw_output at which fixed score is ~0.5).",
    )
    p.add_argument(
        "--fix-scale",
        type=float,
        default=float(DEFAULT_SATURATION_FIX.scale),
        help="Saturation-fix scale parameter (smaller = steeper mapping).",
    )
    p.add_argument(
        "--fix-conf-weight",
        type=float,
        default=float(DEFAULT_SATURATION_FIX.confidence_weight),
        help="Optional saturation-fix confidence weight (0 disables).",
    )

    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    word_counts, _total = load_frequency_list(args.data)
    sorted_by_count = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)
    common_k = min(args.common_k, len(sorted_by_count))
    common_words = [w for w, _ in sorted_by_count[:common_k]]
    real_set = set(word_counts.keys())

    bases = choose_bases(common_words, seed=args.seed, max_bases=min(5000, len(common_words)))
    composed = generate_composed_words(bases=bases, real_set=real_set, n=args.n, seed=args.seed)
    gibberish = generate_gibberish_words(
        lengths=[len(w) for w in composed], real_set=real_set, n=len(composed), seed=args.seed + 1
    )
    common_sample = bases[: min(len(bases), len(composed))]

    model, _ckpt = load_model(args.model, device=device)

    fix_config = SaturationFixConfig(
        eps=float(DEFAULT_SATURATION_FIX.eps),
        center=float(args.fix_center),
        scale=float(args.fix_scale),
        confidence_weight=float(args.fix_conf_weight),
        confidence_center=float(DEFAULT_SATURATION_FIX.confidence_center),
    )

    # Raw predictions
    pred_common = _batch_predict(
        model, common_sample, device=device, saturation_fix=False, fix_config=fix_config
    )
    pred_comp = _batch_predict(
        model, composed, device=device, saturation_fix=False, fix_config=fix_config
    )
    pred_gib = _batch_predict(
        model, gibberish, device=device, saturation_fix=False, fix_config=fix_config
    )

    # Fixed predictions (only changes saturated tail)
    pred_comp_fix = _batch_predict(
        model,
        composed,
        device=device,
        saturation_fix=True,
        fix_config=fix_config,
    )
    pred_gib_fix = _batch_predict(
        model,
        gibberish,
        device=device,
        saturation_fix=True,
        fix_config=fix_config,
    )

    # AUROC: gibberish should be "more rare" than composed.
    y = np.array([0] * len(composed) + [1] * len(gibberish), dtype=int)
    scores_raw = np.concatenate([pred_comp.model, pred_gib.model], axis=0)
    auc_raw = _auc_from_scores(y, scores_raw)

    auc_fix = float("nan")
    if pred_comp_fix.fixed is not None and pred_gib_fix.fixed is not None:
        scores_fix = np.concatenate([pred_comp_fix.fixed, pred_gib_fix.fixed], axis=0)
        auc_fix = _auc_from_scores(y, scores_fix)

    auc_raw_output = float("nan")
    if pred_comp.raw is not None and pred_gib.raw is not None:
        scores_ro = np.concatenate([pred_comp.raw, pred_gib.raw], axis=0)
        auc_raw_output = _auc_from_scores(y, scores_ro)

    auc_confidence = float("nan")
    if pred_comp.confidence is not None and pred_gib.confidence is not None:
        scores_conf = np.concatenate([pred_comp.confidence, pred_gib.confidence], axis=0)
        auc_confidence = _auc_from_scores(y, scores_conf)

    print("\nOOV calibration / saturation evaluation")
    print(f"  model: {args.model}")
    print(f"  data:  {args.data}")
    print(
        f"  n_composed={len(composed)}  n_gibberish={len(gibberish)}  n_common={len(common_sample)}"
    )
    print(
        f"  fix: center={fix_config.center:g}  scale={fix_config.scale:g}  conf_w={fix_config.confidence_weight:g}"
    )
    print("")

    s_common = _summarize_scores("common", pred_common.model)
    s_comp = _summarize_scores("composed", pred_comp.model)
    s_gib = _summarize_scores("gibberish", pred_gib.model)

    print("Raw model outputs")
    print(
        f"  common:    mean={s_common['mean']:.4f} p50={s_common['p50']:.4f} p95={s_common['p95']:.4f} sat@1={s_common['sat_1p0']:.1%}"
    )
    print(
        f"  composed:  mean={s_comp['mean']:.4f} p50={s_comp['p50']:.4f} p95={s_comp['p95']:.4f} sat@1={s_comp['sat_1p0']:.1%}"
    )
    print(
        f"  gibberish: mean={s_gib['mean']:.4f} p50={s_gib['p50']:.4f} p95={s_gib['p95']:.4f} sat@1={s_gib['sat_1p0']:.1%}"
    )
    print(f"  AUROC(gibberish vs composed): {auc_raw:.4f}")
    if not np.isnan(auc_raw_output):
        print(f"  AUROC(raw_output):           {auc_raw_output:.4f}")
    if not np.isnan(auc_confidence):
        print(f"  AUROC(confidence):           {auc_confidence:.4f}")

    if pred_comp_fix.fixed is not None and pred_gib_fix.fixed is not None:
        s_comp_fx = _summarize_scores("composed_fix", pred_comp_fix.fixed)
        s_gib_fx = _summarize_scores("gibberish_fix", pred_gib_fix.fixed)
        applied_rate = (
            float(np.mean(pred_comp_fix.applied))
            if pred_comp_fix.applied is not None
            else float("nan")
        )

        print("\nWith saturation-fix (only affects clamped-high tail)")
        print(
            f"  composed:  mean={s_comp_fx['mean']:.4f} p50={s_comp_fx['p50']:.4f} p95={s_comp_fx['p95']:.4f} sat@1={s_comp_fx['sat_1p0']:.1%} applied={applied_rate:.1%}"
        )
        print(
            f"  gibberish: mean={s_gib_fx['mean']:.4f} p50={s_gib_fx['p50']:.4f} p95={s_gib_fx['p95']:.4f} sat@1={s_gib_fx['sat_1p0']:.1%}"
        )
        print(f"  AUROC(gibberish vs composed): {auc_fix:.4f}")

        # Show a few examples where the fix meaningfully changed the score.
        if pred_comp_fix.fixed is not None and pred_comp_fix.applied is not None:
            deltas = pred_comp_fix.model - pred_comp_fix.fixed
            idx = np.argsort(-deltas)[:8]
            print("\nLargest composed-word adjustments (raw -> fixed)")
            for i in idx:
                if not pred_comp_fix.applied[i]:
                    continue
                print(
                    f"  {composed[i]:20} {pred_comp_fix.model[i]:.4f} -> {pred_comp_fix.fixed[i]:.4f}"
                )


if __name__ == "__main__":
    main()
