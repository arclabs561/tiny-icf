"""Analyze training results and model predictions."""

import sys
from pathlib import Path

import torch
import numpy as np

from tiny_icf.data import compute_normalized_icf, load_frequency_list
from tiny_icf.model import UniversalICF
from tiny_icf.predict import predict_icf


def analyze_model_performance(model_path: str, data_path: str):
    """Analyze model predictions vs ground truth."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ground truth
    word_counts, total_tokens = load_frequency_list(Path(data_path))
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    
    # Load model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Get predictions for all words
    words = list(word_icf.keys())
    predictions = []
    ground_truth = []
    
    print("Analyzing model performance...")
    print(f"Total words: {len(words)}")
    print(f"Total tokens: {total_tokens:,}")
    print("\n" + "=" * 80)
    print("Word-by-word comparison:")
    print("=" * 80)
    print(f"{'Word':<20} {'Ground Truth':<15} {'Predicted':<15} {'Error':<15} {'Status'}")
    print("-" * 80)
    
    for word in sorted(words, key=lambda w: word_icf[w]):
        gt = word_icf[word]
        pred = predict_icf(model, word, device)
        error = abs(pred - gt)
        
        predictions.append(pred)
        ground_truth.append(gt)
        
        status = "✓" if error < 0.1 else "✗"
        print(f"{word:<20} {gt:<15.4f} {pred:<15.4f} {error:<15.4f} {status}")
    
    # Statistics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    mae = np.mean(np.abs(predictions - ground_truth))
    mse = np.mean((predictions - ground_truth) ** 2)
    rmse = np.sqrt(mse)
    
    # Spearman correlation
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(predictions, ground_truth)
    
    print("\n" + "=" * 80)
    print("Performance Statistics:")
    print("=" * 80)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Spearman Correlation: {correlation:.4f} (p={p_value:.4f})")
    print(f"Prediction Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Ground Truth Range: [{ground_truth.min():.4f}, {ground_truth.max():.4f}]")
    
    # Check if model is just predicting the mean
    mean_pred = predictions.mean()
    mean_gt = ground_truth.mean()
    print(f"\nMean Prediction: {mean_pred:.4f}")
    print(f"Mean Ground Truth: {mean_gt:.4f}")
    
    if abs(mean_pred - mean_gt) < 0.05 and predictions.std() < 0.1:
        print("\n⚠️  WARNING: Model appears to be predicting near-constant values!")
        print("   This suggests the model is not learning properly.")
        print("   Possible causes:")
        print("   - Dataset too small")
        print("   - Learning rate too high/low")
        print("   - Model capacity insufficient")
        print("   - Need more training epochs")
    
    # Distribution analysis
    print("\n" + "=" * 80)
    print("Distribution Analysis:")
    print("=" * 80)
    print(f"Ground Truth - Mean: {ground_truth.mean():.4f}, Std: {ground_truth.std():.4f}")
    print(f"Predictions - Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")
    
    # Check ranking
    sorted_gt = sorted(enumerate(ground_truth), key=lambda x: x[1])
    sorted_pred = sorted(enumerate(predictions), key=lambda x: x[1])
    
    # Count how many top-10 words are correctly ranked
    top10_gt_indices = [i for i, _ in sorted_gt[:10]]
    top10_pred_indices = [i for i, _ in sorted_pred[:10]]
    top10_overlap = len(set(top10_gt_indices) & set(top10_pred_indices))
    
    print(f"\nTop-10 Ranking Accuracy: {top10_overlap}/10 words correctly identified")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "p_value": p_value,
        "top10_overlap": top10_overlap,
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "test_frequencies.csv"
    
    analyze_model_performance(model_path, data_path)

