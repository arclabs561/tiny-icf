#!/usr/bin/env python3
"""Train multiple model variations for comparison."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from tiny_icf.model import UniversalICF
from tiny_icf.model_hierarchical import HierarchicalICF, BoxEmbeddingICF
from tiny_icf.data import load_frequency_list, compute_normalized_icf, WordICFDataset, stratified_sample
from tiny_icf.loss import CombinedLoss
from tiny_icf.loss_multi import EnhancedMultiLoss
from tiny_icf.train import train_epoch, validate, generate_ranking_pairs


def train_variation(
    model_class,
    model_name: str,
    data_path: Path,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    use_multi_loss: bool = False,
    output_dir: Path = Path("models/variations"),
):
    """Train a single model variation."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}\n")
    
    # Load data
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    train_dataset = WordICFDataset(train_samples, max_length=20, augment_prob=0.1)
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = model_class()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss
    if use_multi_loss:
        criterion = EnhancedMultiLoss()
    else:
        criterion = CombinedLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    best_val_loss = float('inf')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name.lower()}.pt"
    
    for epoch in range(epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        model.eval()
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ Saved best model (val loss: {val_loss:.4f})")
    
    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_path}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train multiple model variations")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per model")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=Path, default=Path("models/variations"), help="Output directory")
    parser.add_argument("--multi-loss", action="store_true", help="Use enhanced multi-loss")
    
    args = parser.parse_args()
    
    variations = [
        (UniversalICF, "UniversalICF", False),
        (HierarchicalICF, "HierarchicalICF", False),
        (BoxEmbeddingICF, "BoxEmbeddingICF", False),
        (UniversalICF, "UniversalICF_MultiLoss", True),
    ]
    
    results = []
    
    for model_class, model_name, use_multi in variations:
        if args.multi_loss and not use_multi:
            continue  # Skip if only training multi-loss variants
        
        try:
            model, val_loss = train_variation(
                model_class,
                model_name,
                args.data,
                args.epochs,
                args.batch_size,
                args.lr,
                use_multi,
                args.output_dir,
            )
            results.append((model_name, model.count_parameters(), val_loss))
        except Exception as e:
            print(f"✗ Error training {model_name}: {e}")
            results.append((model_name, 0, float('inf')))
    
    # Summary
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"{'Model':<25} {'Params':<12} {'Val Loss':<12}")
    print("-"*70)
    for name, params, loss in results:
        print(f"{name:<25} {params:>10,} {loss:>12.4f}")
    
    # Best model
    best = min(results, key=lambda x: x[2])
    print(f"\n✓ Best model: {best[0]} (val loss: {best[2]:.4f}, {best[1]:,} params)")


if __name__ == "__main__":
    main()

