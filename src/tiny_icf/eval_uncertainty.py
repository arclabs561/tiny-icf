# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24.0",
#   "scipy>=1.10.0",
# ]
# ///
"""
Uncertainty quantification for ICF predictions.

Implements:
- Bootstrap confidence intervals
- Quantile regression for prediction intervals
- Ensemble-based uncertainty (if multiple models available)
- Prediction variance estimation
"""

from typing import Dict, List, Optional
import numpy as np


def bootstrap_confidence_intervals(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = "percentile",
) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals for predictions.

    Uses bootstrap resampling to estimate prediction uncertainty.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: 'percentile' or 'bca' (bias-corrected accelerated)

    Returns:
        Dictionary with 'lower', 'upper', 'mean', 'std' confidence intervals
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    n_samples = len(predictions)

    if n_samples < 10:
        # Too few samples for bootstrap
        return {
            "lower": float(predictions.min()),
            "upper": float(predictions.max()),
            "mean": float(predictions.mean()),
            "std": float(predictions.std()),
        }

    # Bootstrap resampling
    bootstrap_means = []
    bootstrap_errors = []

    alpha = 1.0 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_preds = predictions[indices]
        boot_targets = targets[indices]

        # Compute mean prediction and error
        bootstrap_means.append(boot_preds.mean())
        bootstrap_errors.append(np.mean(np.abs(boot_preds - boot_targets)))

    bootstrap_means = np.array(bootstrap_means)
    bootstrap_errors = np.array(bootstrap_errors)

    if method == "percentile":
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
    else:
        # Simple percentile for now (BCA is more complex)
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

    return {
        "lower": float(ci_lower),
        "upper": float(ci_upper),
        "mean": float(bootstrap_means.mean()),
        "std": float(bootstrap_means.std()),
        "error_mean": float(bootstrap_errors.mean()),
        "error_std": float(bootstrap_errors.std()),
    }


def quantile_regression_intervals(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute prediction intervals using quantile regression approach.

    Estimates prediction intervals by analyzing the distribution of errors
    at different prediction levels.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        quantiles: List of quantiles to compute (default: [0.05, 0.95] for 90% interval)

    Returns:
        Dictionary with quantile-based intervals
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    if quantiles is None:
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    errors = predictions - targets
    abs_errors = np.abs(errors)

    # Compute quantiles of absolute errors
    error_quantiles = np.percentile(abs_errors, [q * 100 for q in quantiles])

    # Estimate prediction intervals
    # For each quantile, estimate the prediction interval width
    results = {}
    for q, eq in zip(quantiles, error_quantiles):
        results[f"error_q{int(q*100)}"] = float(eq)

    # Prediction intervals: prediction ± error_quantile
    results["interval_90_lower"] = float(predictions.mean() - error_quantiles[-1])
    results["interval_90_upper"] = float(predictions.mean() + error_quantiles[-1])
    results["interval_50_lower"] = float(predictions.mean() - error_quantiles[2])  # median
    results["interval_50_upper"] = float(predictions.mean() + error_quantiles[2])

    return results


def prediction_variance_estimation(
    predictions: np.ndarray,
    targets: np.ndarray,
    method: str = "residual",
) -> Dict[str, float]:
    """
    Estimate prediction variance using residual analysis.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        method: 'residual' (analyze residuals) or 'heteroscedastic' (variance varies with prediction)

    Returns:
        Dictionary with variance estimates
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    residuals = predictions - targets
    abs_residuals = np.abs(residuals)

    if method == "residual":
        # Simple residual variance
        variance = np.var(residuals)
        std = np.std(residuals)

        # Variance by prediction bin (heteroscedasticity check)
        n_bins = 5
        bin_edges = np.linspace(predictions.min(), predictions.max(), n_bins + 1)
        bin_variances = []

        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if mask.sum() > 1:
                bin_variances.append(np.var(residuals[mask]))

        heteroscedasticity = np.std(bin_variances) if bin_variances else 0.0

        return {
            "variance": float(variance),
            "std": float(std),
            "heteroscedasticity": float(heteroscedasticity),
            "mean_abs_error": float(np.mean(abs_residuals)),
            "median_abs_error": float(np.median(abs_residuals)),
        }
    else:
        # Heteroscedastic estimation (variance depends on prediction level)
        # Simple approach: bin predictions and estimate variance per bin
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_variances = []
        bin_centers = []

        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if mask.sum() > 1:
                bin_variances.append(np.var(residuals[mask]))
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

        return {
            "variance_mean": float(np.mean(bin_variances)) if bin_variances else 0.0,
            "variance_std": float(np.std(bin_variances)) if bin_variances else 0.0,
            "heteroscedasticity": float(np.std(bin_variances)) if bin_variances else 0.0,
        }


def ensemble_uncertainty(
    model_predictions: List[np.ndarray],
    method: str = "mean_std",
) -> Dict[str, np.ndarray]:
    """
    Compute uncertainty from ensemble of model predictions.

    Args:
        model_predictions: List of prediction arrays from different models [N_models, N_samples]
        method: 'mean_std' (mean and std) or 'quantile' (quantile-based)

    Returns:
        Dictionary with ensemble predictions and uncertainty estimates
    """
    if not model_predictions:
        return {}

    # Stack predictions
    predictions_stack = np.stack(model_predictions, axis=0)  # [N_models, N_samples]

    if method == "mean_std":
        mean_pred = np.mean(predictions_stack, axis=0)
        std_pred = np.std(predictions_stack, axis=0)

        return {
            "mean": mean_pred,
            "std": std_pred,
            "min": np.min(predictions_stack, axis=0),
            "max": np.max(predictions_stack, axis=0),
        }
    else:
        # Quantile-based
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        quantile_preds = np.percentile(predictions_stack, [q * 100 for q in quantiles], axis=0)

        return {
            "q05": quantile_preds[0],
            "q25": quantile_preds[1],
            "q50": quantile_preds[2],  # median
            "q75": quantile_preds[3],
            "q95": quantile_preds[4],
            "mean": np.mean(predictions_stack, axis=0),
        }


def compute_uncertainty_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, any]:
    """
    Compute comprehensive uncertainty quantification metrics.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary with all uncertainty metrics
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    results = {}

    # Bootstrap confidence intervals
    bootstrap_ci = bootstrap_confidence_intervals(
        predictions, targets, n_bootstrap, confidence_level
    )
    results["bootstrap"] = bootstrap_ci

    # Quantile regression intervals
    quantile_intervals = quantile_regression_intervals(predictions, targets)
    results["quantile"] = quantile_intervals

    # Prediction variance
    variance_est = prediction_variance_estimation(predictions, targets)
    results["variance"] = variance_est

    # Coverage analysis (how often true value is within intervals)
    errors = predictions - targets
    abs_errors = np.abs(errors)

    # 90% interval coverage
    interval_90_width = quantile_intervals.get("error_q95", np.percentile(abs_errors, 95))
    coverage_90 = np.mean(abs_errors <= interval_90_width)

    # 50% interval coverage
    interval_50_width = quantile_intervals.get("error_q50", np.percentile(abs_errors, 50))
    coverage_50 = np.mean(abs_errors <= interval_50_width)

    results["coverage"] = {
        "interval_90": float(coverage_90),
        "interval_50": float(coverage_50),
        "expected_90": 0.90,
        "expected_50": 0.50,
    }

    return results
