#!/usr/bin/env -S uv run
"""Benchmark different training configurations to find optimal settings."""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.eval import compute_metrics, evaluate_jabberwocky
from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.train import generate_ranking_pairs, set_seed, train_epoch, validate


def benchmark_config(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
) -> dict:
    """Benchmark a single configuration."""
    set_seed(42)
    
    # Model
    model = UniversalICF().to(device)
    model.init_weights(mean_icf=0.5)  # Use default mean
    
    # Loss
    criterion = CombinedLoss(
        rank_weight=config.get("rank_weight", 2.0),
        rank_margin=config.get("rank_margin", 0.1),
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    
    # Training
    start_time = time.time()
    train_losses = []
    val_losses = []
    metrics_history = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Quick evaluation every 2 epochs
        if (epoch + 1) % 2 == 0:
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for byte_tensors, icf_targets in val_loader:
                    byte_tensors = byte_tensors.to(device)
                    icf_targets = icf_targets.to(device)
                    predictions = model(byte_tensors)
                    all_preds.append(predictions.cpu().numpy())
                    all_targets.append(icf_targets.cpu().numpy())
            
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
            metrics = compute_metrics(preds, targets)
            metrics_history.append({
                "epoch": epoch + 1,
                **metrics,
                "pred_std": float(preds.std()),
                "pred_range": [float(preds.min()), float(preds.max())],
            })
            model.train()
    
    elapsed_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for byte_tensors, icf_targets in val_loader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            predictions = model(byte_tensors)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(icf_targets.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    final_metrics = compute_metrics(preds, targets)
    jabberwocky = evaluate_jabberwocky(model, device)
    
    return {
        "config": config,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_metrics": final_metrics,
        "jabberwocky": jabberwocky,
        "metrics_history": metrics_history,
        "elapsed_time": elapsed_time,
        "prediction_stats": {
            "mean": float(preds.mean()),
            "std": float(preds.std()),
            "min": float(preds.min()),
            "max": float(preds.max()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark training configurations")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per configuration")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples for quick benchmark")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 70)
    print("Training Configuration Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs per config: {args.epochs}")
    print()
    
    # Load data
    print("Loading data...")
    word_counts, total_tokens = load_frequency_list(args.data)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    test_samples = samples[:args.max_samples]
    split_idx = int(len(test_samples) * 0.8)
    train_samples = test_samples[:split_idx]
    val_samples = test_samples[split_idx:]
    
    train_dataset = WordICFDataset(train_samples, max_length=20, augment_prob=0.1)
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Configurations to benchmark
    configurations = [
        {"name": "baseline", "lr": 1e-3, "rank_weight": 2.0, "rank_margin": 0.1},
        {"name": "higher_lr", "lr": 2e-3, "rank_weight": 2.0, "rank_margin": 0.1},
        {"name": "lower_lr", "lr": 5e-4, "rank_weight": 2.0, "rank_margin": 0.1},
        {"name": "stronger_ranking", "lr": 1e-3, "rank_weight": 3.0, "rank_margin": 0.1},
        {"name": "larger_margin", "lr": 1e-3, "rank_weight": 2.0, "rank_margin": 0.15},
        {"name": "very_strong", "lr": 1e-3, "rank_weight": 4.0, "rank_margin": 0.1},
        {"name": "wide_margin", "lr": 1e-3, "rank_weight": 1.5, "rank_margin": 0.2},
    ]
    
    print(f"\nBenchmarking {len(configurations)} configurations...")
    print()
    
    results = []
    for i, config in enumerate(configurations, 1):
        name = config["name"]
        print(f"[{i}/{len(configurations)}] {name}...")
        result = benchmark_config(config, train_loader, val_loader, device, args.epochs)
        results.append(result)
        
        final = result["final_metrics"]
        print(f"  MAE: {final['mae']:.4f}, Spearman: {final['spearman_corr']:.4f}, "
              f"Time: {result['elapsed_time']:.1f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    
    print(f"\n{'Config':<20} {'MAE':<8} {'Spearman':<10} {'Jabberwocky':<15} {'Time':<10}")
    print("-" * 70)
    
    best_spearman = -1
    best_config = None
    
    for result in results:
        config = result["config"]
        final = result["final_metrics"]
        jabberwocky = result["jabberwocky"]
        jabberwocky_str = f"{jabberwocky['passed_count']}/{jabberwocky['total_count']}"
        
        print(f"{config['name']:<20} {final['mae']:<8.4f} {final['spearman_corr']:<10.4f} "
              f"{jabberwocky_str:<15} {result['elapsed_time']:<10.1f}s")
        
        if final['spearman_corr'] > best_spearman:
            best_spearman = final['spearman_corr']
            best_config = config
    
    print(f"\n✓ Best configuration: {best_config['name']}")
    print(f"  LR: {best_config['lr']:.4f}")
    print(f"  Rank weight: {best_config['rank_weight']}")
    print(f"  Rank margin: {best_config['rank_margin']}")
    print(f"  Spearman: {best_spearman:.4f}")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "configurations": [r["config"] for r in results],
                "results": [
                    {
                        "config": r["config"]["name"],
                        "final_metrics": r["final_metrics"],
                        "jabberwocky": r["jabberwocky"],
                        "prediction_stats": r["prediction_stats"],
                        "elapsed_time": r["elapsed_time"],
                    }
                    for r in results
                ],
                "best_config": best_config,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

