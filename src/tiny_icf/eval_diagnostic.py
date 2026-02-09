"""Comprehensive diagnostic evaluation for ICF prediction models.

Provides concrete examples, error analysis, and detailed metrics beyond loss.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


def compute_prediction_distances(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    words: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute various distance metrics between predictions and targets.

    Args:
        predictions: Model predictions (batch_size,)
        targets: Ground truth ICF values (batch_size,)
        words: Optional word strings for per-word analysis

    Returns:
        Dictionary with distance metrics and statistics
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # Absolute errors
    abs_errors = np.abs(predictions_np - targets_np)

    # Squared errors
    squared_errors = (predictions_np - targets_np) ** 2

    # Relative errors (avoid division by zero)
    relative_errors = np.where(
        targets_np != 0, abs_errors / (np.abs(targets_np) + 1e-8), abs_errors
    )

    # Percent close (within 10%, 20%, 50% of target)
    # Refined: Use relative thresholds that account for target magnitude
    non_zero_mask = np.abs(targets_np) > 1e-6
    if non_zero_mask.sum() > 0:
        percent_close_10 = (
            np.mean(abs_errors[non_zero_mask] < 0.1 * np.abs(targets_np[non_zero_mask])) * 100
        )
        percent_close_20 = (
            np.mean(abs_errors[non_zero_mask] < 0.2 * np.abs(targets_np[non_zero_mask])) * 100
        )
        percent_close_50 = (
            np.mean(abs_errors[non_zero_mask] < 0.5 * np.abs(targets_np[non_zero_mask])) * 100
        )
    else:
        percent_close_10 = percent_close_20 = percent_close_50 = 0.0

    # Additional: Percent within 1%, 5% for high-precision analysis
    if non_zero_mask.sum() > 0:
        percent_close_1 = (
            np.mean(abs_errors[non_zero_mask] < 0.01 * np.abs(targets_np[non_zero_mask])) * 100
        )
        percent_close_5 = (
            np.mean(abs_errors[non_zero_mask] < 0.05 * np.abs(targets_np[non_zero_mask])) * 100
        )
    else:
        percent_close_1 = percent_close_5 = 0.0

    # Absolute thresholds
    percent_close_abs_01 = np.mean(abs_errors < 0.01) * 100
    percent_close_abs_05 = np.mean(abs_errors < 0.05) * 100
    percent_close_abs_10 = np.mean(abs_errors < 0.10) * 100

    metrics = {
        "mean_absolute_error": float(np.mean(abs_errors)),
        "median_absolute_error": float(np.median(abs_errors)),
        "mean_squared_error": float(np.mean(squared_errors)),
        "root_mean_squared_error": float(np.sqrt(np.mean(squared_errors))),
        "mean_relative_error": float(np.mean(relative_errors)),
        "median_relative_error": float(np.median(relative_errors)),
        "percent_close_1pct": percent_close_1,
        "percent_close_5pct": percent_close_5,
        "percent_close_10pct": percent_close_10,
        "percent_close_20pct": percent_close_20,
        "percent_close_50pct": percent_close_50,
        "percent_close_abs_01": percent_close_abs_01,
        "percent_close_abs_05": percent_close_abs_05,
        "percent_close_abs_10": percent_close_abs_10,
        "max_absolute_error": float(np.max(abs_errors)),
        "min_absolute_error": float(np.min(abs_errors)),
        "std_absolute_error": float(np.std(abs_errors)),
    }

    # Per-word analysis if words provided
    if words is not None:
        word_errors = list(zip(words, abs_errors, predictions_np, targets_np))
        word_errors.sort(key=lambda x: x[1], reverse=True)  # Sort by error

        metrics["worst_offenders"] = word_errors[:20]  # Top 20 worst
        metrics["best_predictions"] = word_errors[-20:]  # Top 20 best

    return metrics


def find_interesting_cases(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    words: Optional[List[str]] = None,
    threshold_close: float = 0.05,
    threshold_far: float = 0.5,
) -> Dict[str, List[Tuple]]:
    """Find interesting cases: close calls, false positives, worst offenders.

    Args:
        predictions: Model predictions
        targets: Ground truth
        words: Optional word strings
        threshold_close: Threshold for "close" predictions (absolute)
        threshold_far: Threshold for "far" predictions (absolute)

    Returns:
        Dictionary with interesting cases
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    abs_errors = np.abs(predictions_np - targets_np)

    cases = {
        "close_calls": [],  # Predictions very close to target
        "false_positives": [],  # Predicted high ICF but actually low
        "false_negatives": [],  # Predicted low ICF but actually high
        "worst_offenders": [],  # Largest absolute errors
        "ranking_errors": [],  # Cases where ranking order is wrong
    }

    # Close calls (within threshold)
    close_mask = abs_errors < threshold_close
    close_indices = np.where(close_mask)[0]

    for idx in close_indices[:50]:  # Limit to 50 examples
        word = words[idx] if words is not None else f"word_{idx}"
        cases["close_calls"].append(
            (word, float(predictions_np[idx]), float(targets_np[idx]), float(abs_errors[idx]))
        )

    # False positives: predicted high but actually low
    # Refined: Use adaptive thresholds based on distribution
    pred_median = np.median(predictions_np)
    target_median = np.median(targets_np)
    fp_threshold_high = max(0.5, pred_median + 0.2)  # Top half of predictions
    fp_threshold_low = min(0.3, target_median - 0.2)  # Bottom half of targets
    fp_mask = (predictions_np > fp_threshold_high) & (targets_np < fp_threshold_low)
    fp_indices = np.where(fp_mask)[0]

    for idx in fp_indices[:30]:
        word = words[idx] if words is not None else f"word_{idx}"
        cases["false_positives"].append(
            (word, float(predictions_np[idx]), float(targets_np[idx]), float(abs_errors[idx]))
        )

    # False negatives: predicted low but actually high
    # Refined: Use adaptive thresholds
    fn_threshold_low = min(0.3, pred_median - 0.2)  # Bottom half of predictions
    fn_threshold_high = max(0.5, target_median + 0.2)  # Top half of targets
    fn_mask = (predictions_np < fn_threshold_low) & (targets_np > fn_threshold_high)
    fn_indices = np.where(fn_mask)[0]

    for idx in fn_indices[:30]:
        word = words[idx] if words is not None else f"word_{idx}"
        cases["false_negatives"].append(
            (word, float(predictions_np[idx]), float(targets_np[idx]), float(abs_errors[idx]))
        )

    # Worst offenders: largest absolute errors
    worst_indices = np.argsort(abs_errors)[::-1][:30]
    for idx in worst_indices:
        word = words[idx] if words is not None else f"word_{idx}"
        cases["worst_offenders"].append(
            (word, float(predictions_np[idx]), float(targets_np[idx]), float(abs_errors[idx]))
        )

    # Ranking errors: find pairs where order is wrong
    # (predicted order doesn't match target order)
    if len(predictions_np) > 1:
        ranking_errors = find_ranking_errors(predictions_np, targets_np, words)
        cases["ranking_errors"] = ranking_errors[:30]

    return cases


def find_ranking_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    words: Optional[List[str]] = None,
) -> List[Tuple]:
    """Find cases where predicted ranking doesn't match target ranking.

    Returns pairs of words where the ranking order is incorrect.
    """
    ranking_errors = []

    # Get sorted indices by target (ascending = rare to common)
    target_sorted = np.argsort(targets)

    # Find inversions: pairs where target order != pred order
    n = len(predictions)
    for i in range(min(n, 100)):  # Limit to avoid O(n^2)
        for j in range(i + 1, min(n, 100)):
            idx_i = target_sorted[i]
            idx_j = target_sorted[j]

            # Check if prediction order is reversed
            pred_i = predictions[idx_i]
            pred_j = predictions[idx_j]

            if pred_i > pred_j:  # Wrong order!
                word_i = words[idx_i] if words is not None else f"word_{idx_i}"
                word_j = words[idx_j] if words is not None else f"word_{idx_j}"
                ranking_errors.append(
                    (
                        word_i,
                        float(pred_i),
                        float(targets[idx_i]),
                        word_j,
                        float(pred_j),
                        float(targets[idx_j]),
                        float(abs(pred_i - pred_j)),
                    )
                )

    # Sort by prediction difference (largest errors first)
    ranking_errors.sort(key=lambda x: x[6], reverse=True)

    return ranking_errors


def analyze_error_patterns(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    words: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze patterns in prediction errors.

    Groups errors by word characteristics (length, character patterns, etc.)
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    abs_errors = np.abs(predictions_np - targets_np)

    patterns = {}

    if words is not None:
        # Group by word length
        length_groups = defaultdict(list)
        for word, error, pred, target in zip(words, abs_errors, predictions_np, targets_np):
            length_groups[len(word)].append((error, pred, target))

        length_stats = {}
        for length, errors in sorted(length_groups.items()):
            errors_only = [e[0] for e in errors]
            length_stats[length] = {
                "count": len(errors),
                "mean_error": float(np.mean(errors_only)),
                "median_error": float(np.median(errors_only)),
            }
        patterns["by_length"] = length_stats

        # Group by target ICF range
        icf_ranges = {
            "very_rare": (0.0, 0.2),
            "rare": (0.2, 0.4),
            "common": (0.4, 0.6),
            "very_common": (0.6, 1.0),
        }

        range_stats = {}
        for range_name, (low, high) in icf_ranges.items():
            mask = (targets_np >= low) & (targets_np < high)
            if np.any(mask):
                range_errors = abs_errors[mask]
                range_stats[range_name] = {
                    "count": int(np.sum(mask)),
                    "mean_error": float(np.mean(range_errors)),
                    "median_error": float(np.median(range_errors)),
                    "mean_prediction": float(np.mean(predictions_np[mask])),
                    "mean_target": float(np.mean(targets_np[mask])),
                }
        patterns["by_icf_range"] = range_stats

    return patterns


def compute_ranking_quality_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    top_k: int = 10,
) -> Dict[str, float]:
    """Compute ranking-quality metrics (when available).

    This is an optional enhancement that relies on `rank-eval` (via
    `tiny_icf.eval_ranking_metrics`). If the dependency isn't available, we
    return an empty dict rather than failing the diagnostic pipeline.
    """
    try:
        from tiny_icf.eval_ranking_metrics import compute_ranking_metrics
    except Exception:
        return {}

    preds_np = predictions.detach().cpu().numpy().reshape(-1)
    targets_np = targets.detach().cpu().numpy().reshape(-1)

    n = int(preds_np.shape[0])
    if n == 0:
        return {}

    # Include common cutoffs and the requested `top_k`, bounded by `n`.
    k_values = [1, 3, 5, 10, int(top_k)]
    k_values = sorted({k for k in k_values if 1 <= k <= n})
    if not k_values:
        return {}

    return compute_ranking_metrics(preds_np, targets_np, k_values=k_values, use_graded=True)


def compute_diagnostic_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    words: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive diagnostic metrics.

    This is the main entry point that combines all diagnostic analyses.
    """
    diagnostics = {
        "distances": compute_prediction_distances(predictions, targets, words),
        "interesting_cases": find_interesting_cases(predictions, targets, words),
        "error_patterns": analyze_error_patterns(predictions, targets, words),
        "ranking_quality": compute_ranking_quality_metrics(predictions, targets, top_k=100),
    }

    return diagnostics


def format_diagnostic_report(
    diagnostics: Dict[str, Any],
    top_n: int = 10,
) -> str:
    """Format diagnostic metrics as a human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("ICF Prediction Diagnostic Report")
    lines.append("=" * 70)
    lines.append("")

    # Distance metrics
    dist = diagnostics["distances"]
    lines.append("📊 Distance Metrics:")
    lines.append(f"  Mean Absolute Error: {dist['mean_absolute_error']:.4f}")
    lines.append(f"  Median Absolute Error: {dist['median_absolute_error']:.4f}")
    lines.append(f"  Mean Squared Error: {dist['mean_squared_error']:.4f}")
    lines.append(f"  Root Mean Squared Error: {dist['root_mean_squared_error']:.4f}")
    lines.append(f"  Mean Relative Error: {dist['mean_relative_error']:.4f}")
    lines.append("")
    lines.append("📈 Percent Close Predictions:")
    lines.append(f"  Within 10% of target: {dist['percent_close_10pct']:.1f}%")
    lines.append(f"  Within 20% of target: {dist['percent_close_20pct']:.1f}%")
    lines.append(f"  Within 50% of target: {dist['percent_close_50pct']:.1f}%")
    lines.append(f"  Within 0.01 absolute: {dist['percent_close_abs_01']:.1f}%")
    lines.append(f"  Within 0.05 absolute: {dist['percent_close_abs_05']:.1f}%")
    lines.append(f"  Within 0.10 absolute: {dist['percent_close_abs_10']:.1f}%")
    lines.append("")

    # Interesting cases
    cases = diagnostics["interesting_cases"]
    lines.append("🔍 Interesting Cases:")
    lines.append("")

    if cases["worst_offenders"]:
        lines.append(f"  Worst Offenders (top {min(top_n, len(cases['worst_offenders']))}):")
        for word, pred, target, error in cases["worst_offenders"][:top_n]:
            lines.append(f"    {word:20s} pred={pred:.4f} target={target:.4f} error={error:.4f}")
        lines.append("")

    if cases["close_calls"]:
        lines.append(f"  Close Calls (top {min(top_n, len(cases['close_calls']))}):")
        for word, pred, target, error in cases["close_calls"][:top_n]:
            lines.append(f"    {word:20s} pred={pred:.4f} target={target:.4f} error={error:.4f}")
        lines.append("")

    if cases["false_positives"]:
        lines.append(
            f"  False Positives (predicted high, actually low, top {min(top_n, len(cases['false_positives']))}):"
        )
        for word, pred, target, error in cases["false_positives"][:top_n]:
            lines.append(f"    {word:20s} pred={pred:.4f} target={target:.4f} error={error:.4f}")
        lines.append("")

    if cases["false_negatives"]:
        lines.append(
            f"  False Negatives (predicted low, actually high, top {min(top_n, len(cases['false_negatives']))}):"
        )
        for word, pred, target, error in cases["false_negatives"][:top_n]:
            lines.append(f"    {word:20s} pred={pred:.4f} target={target:.4f} error={error:.4f}")
        lines.append("")

    if cases["ranking_errors"]:
        lines.append(f"  Ranking Errors (top {min(top_n, len(cases['ranking_errors']))}):")
        for word1, pred1, target1, word2, pred2, target2, diff in cases["ranking_errors"][:top_n]:
            lines.append(
                f"    {word1:15s} (pred={pred1:.3f}, tgt={target1:.3f}) vs {word2:15s} (pred={pred2:.3f}, tgt={target2:.3f}) [diff={diff:.3f}]"
            )
        lines.append("")

    # Error patterns
    patterns = diagnostics["error_patterns"]
    if "by_icf_range" in patterns:
        lines.append("📊 Error Patterns by ICF Range:")
        for range_name, stats in patterns["by_icf_range"].items():
            lines.append(
                f"  {range_name:15s} (n={stats['count']:4d}): "
                f"mean_err={stats['mean_error']:.4f}, "
                f"mean_pred={stats['mean_prediction']:.4f}, "
                f"mean_tgt={stats['mean_target']:.4f}"
            )
        lines.append("")

    if "by_length" in patterns:
        lines.append("📊 Error Patterns by Word Length:")
        for length in sorted(patterns["by_length"].keys())[:10]:  # Top 10 lengths
            stats = patterns["by_length"][length]
            lines.append(
                f"  length={length:2d} (n={stats['count']:4d}): "
                f"mean_err={stats['mean_error']:.4f}, "
                f"median_err={stats['median_error']:.4f}"
            )
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)
