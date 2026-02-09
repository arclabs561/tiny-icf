"""Training script with mid-epoch evaluation hooks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tiny_icf.data import (
    WordICFDataset,
    compute_normalized_icf,
    load_frequency_list,
    stratified_sample,
)
from tiny_icf.eval import compute_metrics, evaluate_jabberwocky
from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.train import set_seed, train_epoch, validate


def evaluate_mid_training(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    eval_interval: int = 5,
) -> dict | None:
    """Run evaluation during training if at evaluation interval."""
    if epoch % eval_interval != 0:
        return None

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for byte_tensors, icf_targets in dataloader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)

            predictions = model(byte_tensors)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(icf_targets.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(predictions, targets)
    jabberwocky = evaluate_jabberwocky(model, device)

    return {
        "epoch": epoch,
        "metrics": metrics,
        "jabberwocky": jabberwocky,
        "prediction_stats": {
            "mean": float(predictions.mean()),
            "std": float(predictions.std()),
            "min": float(predictions.min()),
            "max": float(predictions.max()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Train with mid-epoch evaluation")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=20, help="Max word length")
    parser.add_argument("--augment-prob", type=float, default=0.1, help="Augmentation probability")
    parser.add_argument(
        "--output", type=Path, default=Path("models/model_with_eval.pt"), help="Output model path"
    )
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--eval-output", type=Path, help="Path to save evaluation results JSON")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--use-scheduler", action="store_true", help="Use learning rate scheduler")

    args = parser.parse_args()

    # Set random seed
    set_seed(42)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print("Random seed: 42 (reproducible)")
    print(f"Eval interval: Every {args.eval_interval} epochs")

    # Load data
    print("Loading frequency list...")
    word_counts, total_tokens = load_frequency_list(args.data)
    print(f"Loaded {len(word_counts)} words, {total_tokens:,} total tokens")

    # Compute ICF
    print("Computing normalized ICF...")
    word_icf = compute_normalized_icf(word_counts, total_tokens)

    # Stratified sampling
    print("Creating stratified sample...")
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    print(f"Sampled {len(samples)} word-ICF pairs")

    # Split train/val (80/20)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Datasets
    train_dataset = WordICFDataset(
        train_samples, max_length=args.max_length, augment_prob=args.augment_prob
    )
    val_dataset = WordICFDataset(val_samples, max_length=args.max_length, augment_prob=0.0)

    # Use larger batch size for better ranking signal (more pairs per batch)
    effective_batch_size = max(args.batch_size, 64)  # Minimum 64 for better ranking
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False)

    # Model with proper initialization
    model = UniversalICF().to(device)
    sample_icf_values = [icf for _, icf in samples[:1000]]
    mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
    model.init_weights(mean_icf=mean_icf)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Initialized with mean ICF bias: {mean_icf:.4f}")

    # Loss and optimizer
    criterion = CombinedLoss()  # Now with improved defaults (rank_weight=2.0, rank_margin=0.1)
    # Use slightly lower learning rate for more stable training with ranking loss
    effective_lr = args.lr * 0.8  # 20% reduction for stability
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )
        print("Using cosine annealing LR scheduler")

    # Training loop with evaluation
    best_val_loss = float("inf")
    eval_history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Mid-training evaluation
        eval_result = evaluate_mid_training(
            model, val_loader, device, epoch + 1, args.eval_interval
        )
        if eval_result:
            metrics = eval_result["metrics"]
            jabberwocky = eval_result["jabberwocky"]
            pred_stats = eval_result["prediction_stats"]

            print(f"\n📊 Mid-Training Evaluation (Epoch {epoch + 1}):")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  Spearman: {metrics['spearman_corr']:.4f}")
            print(
                f"  Jabberwocky: {jabberwocky['passed_count']}/{jabberwocky['total_count']} ({jabberwocky['pass_rate']:.1%})"
            )
            print(
                f"  Predictions: mean={pred_stats['mean']:.4f}, std={pred_stats['std']:.4f}, range=[{pred_stats['min']:.4f}, {pred_stats['max']:.4f}]"
            )

            eval_history.append(eval_result)

        # Update learning rate
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ Saved best model (val loss: {val_loss:.4f})")

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")

    # Save evaluation history
    if args.eval_output and eval_history:
        # Convert numpy arrays to lists for JSON
        for eval_result in eval_history:
            if "metrics" in eval_result:
                # Already converted by compute_metrics
                pass

        with open(args.eval_output, "w") as f:
            json.dump(eval_history, f, indent=2)
        print(f"✓ Evaluation history saved to {args.eval_output}")


if __name__ == "__main__":
    main()
