"""Advanced evaluation utilities: error analysis, per-category metrics, and detailed diagnostics."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    from scipy.stats import spearmanr, pearsonr, kendalltau
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def analyze_errors_by_frequency(
    predictions: np.ndarray,
    targets: np.ndarray,
    frequency_bins: int = 5,
) -> Dict[str, any]:
    """
    Analyze errors by frequency category (common vs rare words).
    
    Args:
        predictions: Model predictions [N]
        targets: Ground truth ICF scores [N]
        frequency_bins: Number of frequency bins to analyze
    
    Returns:
        Dictionary with per-bin error statistics
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Create frequency bins
    bin_edges = np.linspace(0.0, 1.0, frequency_bins + 1)
    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(frequency_bins)]
    
    results = {
        "bins": [],
        "overall": {
            "mae": float(np.mean(np.abs(predictions - targets))),
            "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        },
    }
    
    for i in range(frequency_bins):
        mask = (targets >= bin_edges[i]) & (targets < bin_edges[i + 1])
        if i == frequency_bins - 1:  # Include upper bound in last bin
            mask = (targets >= bin_edges[i]) & (targets <= bin_edges[i + 1])
        
        if mask.sum() == 0:
            continue
        
        bin_preds = predictions[mask]
        bin_targets = targets[mask]
        errors = bin_preds - bin_targets
        
        results["bins"].append({
            "bin": bin_labels[i],
            "n_samples": int(mask.sum()),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_error": float(np.mean(errors)),
            "pred_mean": float(np.mean(bin_preds)),
            "target_mean": float(np.mean(bin_targets)),
        })
    
    return results


def analyze_errors_by_length(
    predictions: np.ndarray,
    targets: np.ndarray,
    words: List[str],
    length_bins: List[int] = [1, 4, 7, 10, 15, 20],
) -> Dict[str, any]:
    """
    Analyze errors by word length.
    
    Args:
        predictions: Model predictions [N]
        targets: Ground truth ICF scores [N]
        words: List of words [N]
        length_bins: Word length boundaries
    
    Returns:
        Dictionary with per-length-bin error statistics
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    word_lengths = np.array([len(word) for word in words])
    
    results = {
        "bins": [],
    }
    
    for i in range(len(length_bins) - 1):
        min_len, max_len = length_bins[i], length_bins[i + 1]
        mask = (word_lengths >= min_len) & (word_lengths < max_len)
        
        if mask.sum() == 0:
            continue
        
        bin_preds = predictions[mask]
        bin_targets = targets[mask]
        errors = bin_preds - bin_targets
        
        results["bins"].append({
            "length_range": f"{min_len}-{max_len-1}",
            "n_samples": int(mask.sum()),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_error": float(np.mean(errors)),
        })
    
    return results


def find_worst_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    words: List[str],
    top_k: int = 20,
) -> List[Dict[str, any]]:
    """
    Find words with worst prediction errors.
    
    Args:
        predictions: Model predictions [N]
        targets: Ground truth ICF scores [N]
        words: List of words [N]
        top_k: Number of worst predictions to return
    
    Returns:
        List of dictionaries with word, prediction, target, error
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    errors = np.abs(predictions - targets)
    worst_indices = np.argsort(errors)[-top_k:][::-1]
    
    worst = []
    for idx in worst_indices:
        worst.append({
            "word": words[idx] if idx < len(words) else f"word_{idx}",
            "prediction": float(predictions[idx]),
            "target": float(targets[idx]),
            "error": float(errors[idx]),
        })
    
    return worst


def analyze_ranking_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    top_k: int = 100,
) -> Dict[str, any]:
    """
    Analyze ranking errors: which words are misranked.
    
    Args:
        predictions: Model predictions [N]
        targets: Ground truth ICF scores [N]
        top_k: Number of top/bottom words to analyze
    
    Returns:
        Dictionary with ranking error analysis
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Get top-k and bottom-k by target
    top_target_indices = np.argsort(targets)[:top_k]  # Most common
    bottom_target_indices = np.argsort(targets)[-top_k:][::-1]  # Most rare
    
    # Get top-k and bottom-k by prediction
    top_pred_indices = np.argsort(predictions)[:top_k]  # Predicted most common
    bottom_pred_indices = np.argsort(predictions)[-top_k:][::-1]  # Predicted most rare
    
    # Compute overlap
    top_overlap = len(set(top_target_indices) & set(top_pred_indices)) / top_k
    bottom_overlap = len(set(bottom_target_indices) & set(bottom_pred_indices)) / top_k
    
    # Compute ranking errors for top/bottom
    top_ranking_errors = []
    for idx in top_target_indices:
        pred_rank = np.sum(predictions < predictions[idx])
        target_rank = np.sum(targets < targets[idx])
        top_ranking_errors.append(abs(pred_rank - target_rank))
    
    bottom_ranking_errors = []
    for idx in bottom_target_indices:
        pred_rank = np.sum(predictions > predictions[idx])
        target_rank = np.sum(targets > targets[idx])
        bottom_ranking_errors.append(abs(pred_rank - target_rank))
    
    return {
        "top_k_overlap": float(top_overlap),
        "bottom_k_overlap": float(bottom_overlap),
        "top_k_mean_rank_error": float(np.mean(top_ranking_errors)),
        "bottom_k_mean_rank_error": float(np.mean(bottom_ranking_errors)),
    }


def comprehensive_evaluation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    words: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Comprehensive evaluation with detailed error analysis.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device for computation
        words: Optional list of words (for length/error analysis)
    
    Returns:
        Dictionary with comprehensive evaluation results
    """
    from tiny_icf.eval import compute_metrics, evaluate_ranking
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_words_list = []
    
    with torch.no_grad():
        for byte_tensors, icf_targets in dataloader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            predictions = model(byte_tensors)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(icf_targets.cpu().numpy())
            
            # Decode words if not provided
            if words is None:
                for byte_tensor in byte_tensors.cpu():
                    word_bytes = byte_tensor.numpy()
                    word = bytes(word_bytes[word_bytes > 0]).decode('utf-8', errors='ignore')
                    all_words_list.append(word)
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    eval_words = words if words else all_words_list[:len(predictions)]
    
    # Basic metrics
    metrics = compute_metrics(predictions, targets)
    ranking_metrics = evaluate_ranking(predictions, targets, top_k=10)
    
    # Advanced analysis
    frequency_analysis = analyze_errors_by_frequency(predictions, targets)
    length_analysis = analyze_errors_by_length(predictions, targets, eval_words)
    worst_predictions = find_worst_predictions(predictions, targets, eval_words, top_k=20)
    ranking_analysis = analyze_ranking_errors(predictions, targets, top_k=100)
    
    return {
        "metrics": metrics,
        "ranking_metrics": ranking_metrics,
        "frequency_analysis": frequency_analysis,
        "length_analysis": length_analysis,
        "worst_predictions": worst_predictions,
        "ranking_analysis": ranking_analysis,
        "predictions": predictions.tolist(),
        "targets": targets.tolist(),
        "words": eval_words,
    }

