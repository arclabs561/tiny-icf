"""Comprehensive evaluation metrics and utilities for tiny-icf."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any

from tiny_icf.oov_calibration import (
    DEFAULT_SATURATION_FIX,
    SaturationFixConfig,
    apply_saturation_fix,
)

try:
    from scipy.stats import spearmanr, pearsonr, kendalltau

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Calibration and stratified evaluation are now imported directly
# (functions are in this file, but we check for scipy dependency)
HAS_CALIBRATION = HAS_SCIPY
HAS_STRATIFIED = HAS_SCIPY

# RBO metrics are now in this file (consolidated from eval_rbo.py)
try:
    from tiny_icf.eval_rbo import compute_rbo_metrics

    HAS_RBO = True
except ImportError:
    HAS_RBO = False

    def compute_rbo_metrics(*args, **kwargs):
        return {}


# Uncertainty and robustness are imported from separate modules
# (they're substantial enough to keep separate for now)

try:
    from tiny_icf.eval_uncertainty import compute_uncertainty_metrics

    HAS_UNCERTAINTY = True
except ImportError:
    HAS_UNCERTAINTY = False


def compute_metrics(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    assert len(predictions) == len(targets), "Predictions and targets must have same length"

    metrics = {}

    # Absolute errors
    errors = predictions - targets
    abs_errors = np.abs(errors)

    metrics["mae"] = float(np.mean(abs_errors))
    metrics["rmse"] = float(np.sqrt(np.mean(errors**2)))
    metrics["median_ae"] = float(np.median(abs_errors))
    metrics["max_ae"] = float(np.max(abs_errors))
    metrics["p95_ae"] = float(np.percentile(abs_errors, 95))

    # Relative errors (for non-zero targets)
    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        rel_errors = abs_errors[non_zero_mask] / (targets[non_zero_mask] + 1e-8)
        metrics["mre"] = float(np.mean(rel_errors))  # Mean Relative Error
        metrics["median_re"] = float(np.median(rel_errors))
    else:
        metrics["mre"] = 0.0
        metrics["median_re"] = 0.0

    # Correlation metrics with safe NaN handling
    # Research: NaN occurs when predictions/targets have zero variance (constant values)
    # This is common early in training when model hasn't learned to differentiate
    if HAS_SCIPY and len(predictions) > 1:
        # Check for zero variance before computing correlation
        pred_std = np.std(predictions)
        target_std = np.std(targets)

        if pred_std < 1e-8 or target_std < 1e-8:
            # Zero variance: correlation is undefined, return 0.0
            metrics["spearman_corr"] = 0.0
            metrics["spearman_p"] = 1.0
            metrics["pearson_corr"] = 0.0
            metrics["pearson_p"] = 1.0
            metrics["kendall_corr"] = 0.0
            metrics["kendall_p"] = 1.0
        else:
            # Compute correlations with NaN handling
            spearman_corr, spearman_p = spearmanr(predictions, targets)
            pearson_corr, pearson_p = pearsonr(predictions, targets)
            kendall_corr, kendall_p = kendalltau(predictions, targets)

            # Handle NaN/Inf values (can occur with numerical instability)
            metrics["spearman_corr"] = (
                float(spearman_corr)
                if not (np.isnan(spearman_corr) or np.isinf(spearman_corr))
                else 0.0
            )
            metrics["spearman_p"] = (
                float(spearman_p) if not (np.isnan(spearman_p) or np.isinf(spearman_p)) else 1.0
            )
            metrics["pearson_corr"] = (
                float(pearson_corr)
                if not (np.isnan(pearson_corr) or np.isinf(pearson_corr))
                else 0.0
            )
            metrics["pearson_p"] = (
                float(pearson_p) if not (np.isnan(pearson_p) or np.isinf(pearson_p)) else 1.0
            )
            metrics["kendall_corr"] = (
                float(kendall_corr)
                if not (np.isnan(kendall_corr) or np.isinf(kendall_corr))
                else 0.0
            )
            metrics["kendall_p"] = (
                float(kendall_p) if not (np.isnan(kendall_p) or np.isinf(kendall_p)) else 1.0
            )
    else:
        metrics["spearman_corr"] = 0.0
        metrics["spearman_p"] = 1.0
        metrics["pearson_corr"] = 0.0
        metrics["pearson_p"] = 1.0
        metrics["kendall_corr"] = 0.0
        metrics["kendall_p"] = 1.0

    # Prediction statistics
    metrics["pred_min"] = float(predictions.min())
    metrics["pred_max"] = float(predictions.max())
    metrics["pred_mean"] = float(predictions.mean())
    metrics["pred_std"] = float(predictions.std())
    metrics["pred_median"] = float(np.median(predictions))
    metrics["pred_q25"] = float(np.percentile(predictions, 25))
    metrics["pred_q75"] = float(np.percentile(predictions, 75))
    metrics["pred_iqr"] = float(np.percentile(predictions, 75) - np.percentile(predictions, 25))

    # Target statistics
    metrics["target_min"] = float(targets.min())
    metrics["target_max"] = float(targets.max())
    metrics["target_mean"] = float(targets.mean())
    metrics["target_std"] = float(targets.std())
    metrics["target_median"] = float(np.median(targets))
    metrics["target_q25"] = float(np.percentile(targets, 25))
    metrics["target_q75"] = float(np.percentile(targets, 75))
    metrics["target_iqr"] = float(np.percentile(targets, 75) - np.percentile(targets, 25))

    # Distribution similarity
    metrics["mean_diff"] = float(abs(predictions.mean() - targets.mean()))
    metrics["std_ratio"] = float(predictions.std() / (targets.std() + 1e-8))
    metrics["median_diff"] = float(abs(np.median(predictions) - np.median(targets)))
    metrics["iqr_ratio"] = float(metrics["pred_iqr"] / (metrics["target_iqr"] + 1e-8))

    # Distribution overlap (using IQR overlap)
    pred_q25, pred_q75 = metrics["pred_q25"], metrics["pred_q75"]
    target_q25, target_q75 = metrics["target_q25"], metrics["target_q75"]
    overlap_start = max(pred_q25, target_q25)
    overlap_end = min(pred_q75, target_q75)
    overlap_size = max(0, overlap_end - overlap_start)
    total_span = max(pred_q75, target_q75) - min(pred_q25, target_q25)
    metrics["iqr_overlap"] = float(overlap_size / (total_span + 1e-8))

    # Range coverage
    metrics["range_coverage"] = float(
        (predictions.max() - predictions.min()) / (targets.max() - targets.min() + 1e-8)
    )

    # Calibration metrics (now in this file)
    if HAS_CALIBRATION:
        try:
            calib_metrics = compute_calibration_metrics(predictions, targets, n_bins=10)
            metrics.update(calib_metrics)
            # Backward-compatible alias used by older scripts.
            # Prefer ECE when available.
            metrics["calibration_error"] = float(metrics.get("ece", 0.0))
        except Exception:
            # Fallback to simple calibration error
            metrics["calibration_error"] = expected_calibration_error(
                predictions, targets, n_bins=10
            )
    else:
        # Simple calibration error (fallback)
        metrics["calibration_error"] = expected_calibration_error(predictions, targets, n_bins=10)

    # RBO (Rank-Biased Overlap) metrics - emphasizes top-ranked items
    if HAS_RBO:
        try:
            rbo_metrics = compute_rbo_metrics(
                torch.tensor(predictions),
                torch.tensor(targets),
                top_k_values=[10, 50, 100],
            )
            metrics.update(rbo_metrics)
        except Exception:
            # If RBO computation fails, continue without it
            pass

    return metrics


def evaluate_ranking(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate ranking quality (top-K accuracy, etc.).

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        top_k: Number of top items to consider

    Returns:
        Dictionary of ranking metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    # Sort by target (descending: highest ICF first)
    sorted_indices = np.argsort(targets)[::-1]
    top_k_indices_gt = sorted_indices[:top_k]

    # Sort by predictions (descending)
    sorted_indices_pred = np.argsort(predictions)[::-1]
    top_k_indices_pred = sorted_indices_pred[:top_k]

    # Overlap
    overlap = len(set(top_k_indices_gt) & set(top_k_indices_pred))
    precision_at_k = overlap / top_k

    # Ranking position errors
    position_errors = []
    for idx in top_k_indices_gt:
        if idx in sorted_indices_pred:
            pred_rank = np.where(sorted_indices_pred == idx)[0][0]
            gt_rank = np.where(sorted_indices == idx)[0][0]
            position_errors.append(abs(pred_rank - gt_rank))

    metrics = {
        "precision_at_k": float(precision_at_k),
        "top_k_overlap": int(overlap),
        "mean_rank_error": float(np.mean(position_errors)) if position_errors else 0.0,
        "median_rank_error": float(np.median(position_errors)) if position_errors else 0.0,
    }

    return metrics


def evaluate_jabberwocky(
    model: torch.nn.Module,
    device: torch.device,
    test_cases: Optional[List[Tuple[str, float, float, str]]] = None,
    *,
    saturation_fix: bool = False,
    saturation_fix_config: SaturationFixConfig = DEFAULT_SATURATION_FIX,
    calibration: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on Jabberwocky Protocol (pseudo-words).

    Args:
        model: Trained model
        device: Device for computation
        test_cases: List of (word, min_icf, max_icf, description) tuples

    Returns:
        Dictionary with results and pass rate
    """
    if test_cases is None:
        # Bounds calibrated to per-language log-ICF targets from word_frequency_multilingual.csv:
        #   en:the=0.140  en:and=0.175  en:is=0.213
        #   en:table=0.438  en:running=0.418  en:beautiful=0.419
        #   en:xylophone=0.736  en:soliloquy=0.727
        test_cases = [
            # ── very common ──────────────────────────────────────────────────
            ("the", 0.00, 0.22, "Most-common English word (ICF≈0.14)"),
            ("and", 0.00, 0.25, "2nd-most-common English word (ICF≈0.18)"),
            ("is", 0.05, 0.30, "Very common copula (ICF≈0.21)"),
            # ── moderately common ────────────────────────────────────────────
            ("table", 0.30, 0.60, "Common concrete noun (ICF≈0.44)"),
            ("running", 0.28, 0.58, "Common verb form (ICF≈0.42)"),
            ("beautiful", 0.30, 0.58, "Moderate-frequency adjective (ICF≈0.42)"),
            # ── rare real words ──────────────────────────────────────────────
            ("xylophone", 0.62, 0.90, "Rare but valid word (ICF≈0.74)"),
            ("soliloquy", 0.60, 0.90, "Rare literary word (ICF≈0.73)"),
            # ── plausible OOV (composed / looks English) ────────────────────
            ("flimjam", 0.55, 0.88, "Nonsense but looks English"),
            ("unfriendliness", 0.35, 0.72, "Long composed word"),
            ("preunification", 0.55, 0.90, "Plausible derived form"),
            # ── structural gibberish ─────────────────────────────────────────
            ("qzxbjk", 0.90, 1.00, "Impossible consonant cluster"),
            ("zzzzz", 0.90, 1.00, "Repeated impossible letter"),
        ]

    model.eval()
    results = []

    for word, min_icf, max_icf, description in test_cases:
        # Convert word to bytes (character-boundary safe)
        from tiny_icf.predict import word_to_bytes

        byte_tensor = word_to_bytes(word, max_length=20).to(device)

        # Predict (optionally with saturation fix)
        with torch.no_grad():
            icf_model = float(model(byte_tensor).item())
            icf = icf_model
            raw_output = None
            applied = False
            if saturation_fix:
                try:
                    pred, feats = model(byte_tensor, return_features=True)
                    icf_model = float(pred.squeeze().item())
                    raw_output = float(feats.get("raw_output", pred).squeeze().item())
                    confidence = float(
                        feats.get("confidence", pred.new_tensor(0.5)).squeeze().item()
                    )
                    icf, applied = apply_saturation_fix(
                        icf_score=icf_model,
                        raw_output=raw_output,
                        confidence=confidence,
                        config=saturation_fix_config,
                    )
                except Exception:
                    # Fall back to plain model output.
                    raw_output = None
                    applied = False

        if calibration is not None:
            a, b = calibration
            icf = max(0.0, min(1.0, a + b * icf))

        passed = min_icf <= icf <= max_icf
        results.append(
            {
                "word": word,
                "predicted": icf,
                "predicted_model": icf_model,
                "raw_output": raw_output,
                "saturation_fix_applied": bool(applied),
                "min_icf": min_icf,
                "max_icf": max_icf,
                "passed": passed,
                "description": description,
            }
        )

    passed_count = sum(1 for r in results if r["passed"])
    pass_rate = passed_count / len(results)

    return {
        "pass_rate": pass_rate,
        "passed_count": passed_count,
        "total_count": len(results),
        "results": results,
    }


def evaluate_on_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    max_samples: Optional[int] = None,
    batch_size: int = 64,
    calibration: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        device: Device for computation
        max_samples: Maximum number of samples to evaluate (None = all)
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_predictions = []
    all_targets = []
    all_words = []

    sample_count = 0
    with torch.no_grad():
        for byte_tensors, icf_targets in dataloader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)

            predictions = model(byte_tensors)

            all_predictions.append(predictions.cpu())
            all_targets.append(icf_targets.cpu())

            # Decode words (approximate)
            for byte_tensor in byte_tensors.cpu():
                # NOTE: `bytes(np_array)` uses the buffer protocol (raw memory),
                # which is wrong for int64 tensors (it inserts NUL bytes).
                # We must convert to a true byte sequence first.
                word_bytes = byte_tensor.numpy()
                word_bytes_u8 = word_bytes[word_bytes > 0].astype(np.uint8, copy=False)
                word = word_bytes_u8.tobytes().decode("utf-8", errors="ignore")
                all_words.append(word)

            sample_count += len(byte_tensors)
            if max_samples and sample_count >= max_samples:
                break

    # Concatenate
    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()

    # Optional learned affine calibration (a + b * pred, clip to [0,1])
    if calibration is not None:
        a, b = calibration
        predictions = np.clip(a + b * predictions.astype(np.float64), 0.0, 1.0).astype(
            predictions.dtype
        )

    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    ranking_metrics = evaluate_ranking(predictions, targets, top_k=10)

    # Stratified evaluation (now in this file)
    stratified_results: Dict[str, Any] = {}
    if HAS_STRATIFIED:
        try:
            stratified_results = stratified_evaluation(predictions, targets)
            rarity_results = evaluate_by_rarity_category(predictions, targets)
            if isinstance(stratified_results, dict):
                stratified_results["by_category"] = rarity_results
        except Exception:
            pass

    # Uncertainty quantification (if available)
    uncertainty_results = {}
    if HAS_UNCERTAINTY:
        try:
            uncertainty_results = compute_uncertainty_metrics(predictions, targets)
        except Exception:
            pass

    return {
        "metrics": metrics,
        "ranking_metrics": ranking_metrics,
        "stratified": stratified_results,
        "uncertainty": uncertainty_results,
        "predictions": predictions,
        "targets": targets,
        "words": all_words[: len(predictions)],
        "n_samples": len(predictions),
    }


# ============================================================================
# Calibration Metrics (consolidated from eval_calibration.py)
# ============================================================================


def expected_calibration_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well-calibrated predictions are by binning predictions
    and comparing mean predicted vs mean observed values in each bin.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        n_bins: Number of bins for calibration

    Returns:
        ECE score (lower is better, 0 = perfectly calibrated)
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(predictions)

    for i in range(n_bins):
        mask = bin_indices == i
        n_samples = mask.sum()

        if n_samples > 0:
            # Mean predicted value in this bin
            pred_mean = predictions[mask].mean()
            # Mean observed value in this bin
            target_mean = targets[mask].mean()
            # Calibration error for this bin
            bin_error = abs(pred_mean - target_mean)
            # Weight by number of samples
            ece += (n_samples / total_samples) * bin_error

    return float(ece)


def maximum_calibration_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the maximum calibration error across all bins.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        n_bins: Number of bins for calibration

    Returns:
        MCE score (lower is better)
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    max_error = 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        n_samples = mask.sum()

        if n_samples > 0:
            pred_mean = predictions[mask].mean()
            target_mean = targets[mask].mean()
            bin_error = abs(pred_mean - target_mean)
            max_error = max(max_error, bin_error)

    return float(max_error)


def brier_score(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Compute Brier score (mean squared error).

    Brier score measures calibration quality for probabilistic predictions.
    Lower is better.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]

    Returns:
        Brier score
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    return float(np.mean((predictions - targets) ** 2))


def compute_calibration_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute all calibration metrics.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with ECE, MCE, Brier score
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    return {
        "ece": expected_calibration_error(predictions, targets, n_bins),
        "mce": maximum_calibration_error(predictions, targets, n_bins),
        "brier_score": brier_score(predictions, targets),
    }


# ============================================================================
# Stratified Evaluation (consolidated from eval_stratified.py)
# ============================================================================


def stratified_evaluation(
    predictions: np.ndarray,
    targets: np.ndarray,
    bins: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance stratified by target ICF bins.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth ICF scores [N]
        bins: List of (min, max) tuples for ICF bins. Default: deciles

    Returns:
        Dictionary mapping bin name to metrics
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    if bins is None:
        # Default: deciles (0-0.1, 0.1-0.2, ..., 0.9-1.0)
        bins = [(i / 10, (i + 1) / 10) for i in range(10)]
        bin_names = [f"decile_{i+1}" for i in range(10)]
    else:
        bin_names = [f"bin_{i+1}" for i in range(len(bins))]

    results = {}

    for (min_val, max_val), bin_name in zip(bins, bin_names):
        # Filter samples in this bin
        mask = (targets >= min_val) & (targets < max_val)
        if max_val >= 1.0:  # Include upper bound for last bin
            mask = (targets >= min_val) & (targets <= max_val)

        bin_predictions = predictions[mask]
        bin_targets = targets[mask]

        if len(bin_predictions) == 0:
            results[bin_name] = {
                "n_samples": 0,
                "spearman": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
            }
            continue

        # Compute metrics for this bin
        mae = np.mean(np.abs(bin_predictions - bin_targets))
        rmse = np.sqrt(np.mean((bin_predictions - bin_targets) ** 2))

        if HAS_SCIPY and len(bin_predictions) > 1:
            spearman, _ = spearmanr(bin_predictions, bin_targets)
            if np.isnan(spearman):
                spearman = 0.0
        else:
            spearman = 0.0

        results[bin_name] = {
            "n_samples": int(mask.sum()),
            "spearman": float(spearman),
            "mae": float(mae),
            "rmse": float(rmse),
            "target_mean": float(bin_targets.mean()),
            "pred_mean": float(bin_predictions.mean()),
            "target_std": float(bin_targets.std()),
            "pred_std": float(bin_predictions.std()),
        }

    return results


def evaluate_by_rarity_category(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate by semantic rarity categories.

    Categories:
    - Common (0.0-0.1): Very common words
    - Uncommon (0.1-0.3): Less common but still frequent
    - Rare (0.3-0.7): Rare words
    - Very rare (0.7-1.0): Very rare or gibberish

    Args:
        predictions: Model predictions [N]
        targets: Ground truth ICF scores [N]

    Returns:
        Dictionary mapping category name to metrics
    """
    bins = [
        (0.0, 0.1, "common"),
        (0.1, 0.3, "uncommon"),
        (0.3, 0.7, "rare"),
        (0.7, 1.0, "very_rare"),
    ]

    results = {}

    for min_val, max_val, name in bins:
        mask = (targets >= min_val) & (targets <= max_val)

        bin_predictions = predictions[mask]
        bin_targets = targets[mask]

        if len(bin_predictions) == 0:
            results[name] = {
                "n_samples": 0,
                "spearman": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
            }
            continue

        mae = np.mean(np.abs(bin_predictions - bin_targets))
        rmse = np.sqrt(np.mean((bin_predictions - bin_targets) ** 2))

        if HAS_SCIPY and len(bin_predictions) > 1:
            spearman, _ = spearmanr(bin_predictions, bin_targets)
            if np.isnan(spearman):
                spearman = 0.0
        else:
            spearman = 0.0

        results[name] = {
            "n_samples": int(mask.sum()),
            "spearman": float(spearman),
            "mae": float(mae),
            "rmse": float(rmse),
            "target_mean": float(bin_targets.mean()),
            "pred_mean": float(bin_predictions.mean()),
        }

    return results
