#!/usr/bin/env -S uv run
"""Visualize training history and model predictions."""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, using text-only mode")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def plot_training_history(history_file: Path, output: Path | None = None):
    """Plot training history from JSON file."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot plot")
        return
    
    with open(history_file, "r") as f:
        history = json.load(f)
    
    epochs = [m["epoch"] for m in history.get("metrics", [])]
    if not epochs:
        print("No metrics found in history file")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss
    if "train_loss" in history and "val_loss" in history:
        train_epochs = list(range(1, len(history["train_loss"]) + 1))
        val_epochs = list(range(1, len(history["val_loss"]) + 1))
        axes[0, 0].plot(train_epochs, history["train_loss"], "r-", label="Train", linewidth=2)
        axes[0, 0].plot(val_epochs, history["val_loss"], "b-", label="Val", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # MAE
    if history.get("metrics"):
        maes = [m["mae"] for m in history["metrics"]]
        axes[0, 1].plot(epochs, maes, "g-", linewidth=2)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].set_title("Mean Absolute Error")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Spearman
    if history.get("metrics"):
        spearmans = [m["spearman_corr"] for m in history["metrics"]]
        axes[0, 2].plot(epochs, spearmans, "m-", linewidth=2)
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Spearman Correlation")
        axes[0, 2].set_title("Spearman Correlation")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=0.8, color="r", linestyle="--", alpha=0.5, label="Target")
        axes[0, 2].legend()
    
    # Prediction std
    if history.get("metrics"):
        pred_stds = [m.get("prediction_stats", {}).get("std", 0) for m in history["metrics"]]
        axes[1, 0].plot(epochs, pred_stds, "c-", linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Prediction Std")
        axes[1, 0].set_title("Prediction Standard Deviation")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.05, color="r", linestyle="--", alpha=0.5, label="Target")
        axes[1, 0].legend()
    
    # Learning rate
    if "learning_rates" in history:
        lr_epochs = list(range(1, len(history["learning_rates"]) + 1))
        axes[1, 1].plot(lr_epochs, history["learning_rates"], "orange", linewidth=2)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)
    
    # Jabberwocky pass rate
    if history.get("metrics"):
        jabberwocky_rates = [
            m.get("jabberwocky", {}).get("pass_rate", 0) for m in history["metrics"]
        ]
        axes[1, 2].plot(epochs, jabberwocky_rates, "purple", linewidth=2, marker="o")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Pass Rate")
        axes[1, 2].set_title("Jabberwocky Protocol")
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=0.8, color="g", linestyle="--", alpha=0.5, label="Target")
        axes[1, 2].legend()
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output}")
    else:
        plt.show()


def plot_predictions_vs_targets(model_path: Path, data_path: Path, output: Path | None = None):
    """Plot predictions vs targets scatter plot."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot plot")
        return
    
    import torch
    from torch.utils.data import DataLoader
    
    from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
    from tiny_icf.model import UniversalICF
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    test_samples = samples[:2000]
    dataset = WordICFDataset(test_samples, max_length=20, augment_prob=0.0)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Get predictions
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for byte_tensors, icf_targets in dataloader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            predictions = model(byte_tensors)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(icf_targets.cpu().numpy())
    
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    axes[0].scatter(targets, preds, alpha=0.3, s=1)
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect")
    axes[0].set_xlabel("Target ICF")
    axes[0].set_ylabel("Predicted ICF")
    axes[0].set_title("Predictions vs Targets")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Residual plot
    residuals = preds - targets
    axes[1].scatter(targets, residuals, alpha=0.3, s=1)
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Target ICF")
    axes[1].set_ylabel("Residual (Pred - Target)")
    axes[1].set_title("Residual Plot")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize training history and predictions")
    parser.add_argument("--history", type=Path, help="Training history JSON file")
    parser.add_argument("--model", type=Path, help="Trained model file")
    parser.add_argument("--data", type=Path, help="Data CSV file (for prediction plots)")
    parser.add_argument("--output", type=Path, help="Output plot file")
    parser.add_argument("--type", type=str, choices=["history", "predictions", "both"], 
                       default="history", help="Type of visualization")
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    if args.type in ["history", "both"]:
        if args.history:
            plot_training_history(args.history, args.output)
        else:
            print("--history required for history plots")
    
    if args.type in ["predictions", "both"]:
        if args.model and args.data:
            output = args.output or Path("predictions_plot.png")
            plot_predictions_vs_targets(args.model, args.data, output)
        else:
            print("--model and --data required for prediction plots")


if __name__ == "__main__":
    main()

