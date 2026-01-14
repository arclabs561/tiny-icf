#!/usr/bin/env -S uv run
"""
All-in-one training script that:
1. Validates data
2. Trains model with best practices
3. Evaluates comprehensively
4. Exports model
5. Generates report
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.eval import compute_metrics, evaluate_jabberwocky
from tiny_icf.eval_advanced import comprehensive_evaluation
from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.scheduler import AdaptiveCosineAnnealingLR
from tiny_icf.train import generate_ranking_pairs, set_seed, train_epoch, validate


def main():
    parser = argparse.ArgumentParser(
        description="All-in-one training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=Path, default=Path("models/model_all_in_one.pt"), help="Output model")
    parser.add_argument("--report", type=Path, default=Path("training_report.json"), help="Training report")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer epochs, smaller dataset)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("All-in-One Training Pipeline")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Quick mode: {args.quick}")
    print()
    
    set_seed(42)
    
    if args.device == "auto":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        import torch
        device = torch.device(args.device)
    
    # Step 1: Load and validate data
    print("Step 1: Loading data...")
    word_counts, total_tokens = load_frequency_list(args.data)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    if args.quick:
        samples = samples[:5000]
    
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Step 2: Prepare datasets
    print("\nStep 2: Preparing datasets...")
    train_dataset = WordICFDataset(train_samples, max_length=20, augment_prob=0.1)
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Step 3: Initialize model
    print("\nStep 3: Initializing model...")
    model = UniversalICF().to(device)
    sample_icf_values = [icf for _, icf in train_samples[:1000]]
    mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
    model.init_weights(mean_icf=mean_icf)
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Step 4: Setup training
    print("\nStep 4: Setting up training...")
    criterion = CombinedLoss()
    import torch
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = args.lr
    
    scheduler = AdaptiveCosineAnnealingLR(
        optimizer, T_max=args.epochs // 3, eta_min=1e-5,
        metric="spearman_corr", mode="max", patience=5
    )
    
    # Step 5: Training
    print(f"\nStep 5: Training ({args.epochs} epochs)...")
    history = {
        "train_loss": [],
        "val_loss": [],
        "metrics": [],
    }
    
    best_spearman = -1
    best_model_state = None
    
    from tqdm import tqdm
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
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
            
            import numpy as np
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
            metrics = compute_metrics(preds, targets)
            
            print(f"  Epoch {epoch+1}/{args.epochs}: MAE={metrics['mae']:.4f}, Spearman={metrics['spearman_corr']:.4f}")
            
            if metrics['spearman_corr'] > best_spearman:
                best_spearman = metrics['spearman_corr']
                best_model_state = model.state_dict().copy()
            
            history["metrics"].append({
                "epoch": epoch + 1,
                **metrics,
            })
            
            scheduler.step(metrics, epoch=epoch)
            model.train()
    
    # Step 6: Final evaluation
    print("\nStep 6: Final evaluation...")
    if best_model_state:
        model.load_state_dict(best_model_state)
    
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
    
    import numpy as np
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    final_metrics = compute_metrics(preds, targets)
    jabberwocky = evaluate_jabberwocky(model, device)
    
    print(f"  MAE: {final_metrics['mae']:.4f}")
    print(f"  Spearman: {final_metrics['spearman_corr']:.4f}")
    print(f"  Jabberwocky: {jabberwocky['passed_count']}/{jabberwocky['total_count']}")
    
    # Step 7: Save model
    print(f"\nStep 7: Saving model...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"  Saved to {args.output}")
    
    # Step 8: Generate report
    print(f"\nStep 8: Generating report...")
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "data_file": str(args.data),
        },
        "final_metrics": final_metrics,
        "jabberwocky": jabberwocky,
        "prediction_stats": {
            "mean": float(preds.mean()),
            "std": float(preds.std()),
            "min": float(preds.min()),
            "max": float(preds.max()),
        },
        "training_history": history,
        "best_epoch": best_spearman,
    }
    
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {args.report}")
    
    print("\n" + "=" * 70)
    print("✅ All-in-One Training Complete!")
    print("=" * 70)
    print(f"Model: {args.output}")
    print(f"Report: {args.report}")
    print(f"Best Spearman: {best_spearman:.4f}")


if __name__ == "__main__":
    main()

