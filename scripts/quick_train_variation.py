#!/usr/bin/env -S uv run
"""Quick training test for a single model variation (for validation)."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_icf.model import UniversalICF
from tiny_icf.model_hierarchical import HierarchicalICF, BoxEmbeddingICF
from tiny_icf.nano_model import NanoICF
from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.loss import CombinedLoss
from tiny_icf.train import generate_ranking_pairs, set_seed, validate


def quick_train(
    model_class,
    model_name: str,
    data_path: Path,
    epochs: int = 10,
    batch_size: int = 64,
    max_samples: int = 5000,
    output_path: Path | None = None,
):
    """Quick training test for a model variant."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"Quick Training: {model_name}")
    print(f"{'='*70}\n")
    
    # Load data (small subset)
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    
    # Use small subset for quick test
    test_samples = samples[:max_samples]
    split_idx = int(len(test_samples) * 0.8)
    train_samples = test_samples[:split_idx]
    val_samples = test_samples[split_idx:]
    
    train_dataset = WordICFDataset(train_samples, max_length=20, augment_prob=0.1)
    val_dataset = WordICFDataset(val_samples, max_length=20, augment_prob=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model with proper initialization
    model = model_class().to(device)
    sample_icf_values = [icf for _, icf in train_samples[:1000]]
    mean_icf = sum(sample_icf_values) / len(sample_icf_values) if sample_icf_values else 0.4
    
    # Initialize if method exists
    if hasattr(model, 'init_weights'):
        model.init_weights(mean_icf=mean_icf)
    else:
        # Fallback initialization
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    print(f"Model: {model_name}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Initialized with mean ICF: {mean_icf:.4f}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Epochs: {epochs}\n")
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for byte_tensors, icf_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(byte_tensors)
            
            # Generate ranking pairs
            n_pairs = len(icf_targets) // 2
            pairs = generate_ranking_pairs(icf_targets, n_pairs)
            
            loss = criterion(predictions, icf_targets, pairs=pairs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / n_batches
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"  Train loss: {avg_train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_path)
                print(f"  ✓ Saved best model (val loss: {val_loss:.4f})")
    
    print(f"\n✓ Training complete. Best validation loss: {best_val_loss:.4f}")
    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Quick training test for model variant")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["universal", "hierarchical", "box", "nano"],
                       help="Model variant to train")
    parser.add_argument("--data", type=Path, default=Path("data/word_frequency.csv"),
                       help="Path to frequency CSV")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples for quick test")
    parser.add_argument("--output", type=Path, help="Output model path")
    
    args = parser.parse_args()
    
    # Map model names to classes
    model_map = {
        "universal": UniversalICF,
        "hierarchical": HierarchicalICF,
        "box": BoxEmbeddingICF,
        "nano": NanoICF,
    }
    
    model_class = model_map[args.model]
    model_name = args.model.capitalize() + "ICF"
    
    output_path = args.output or Path(f"models/quick_{args.model}.pt")
    
    quick_train(
        model_class=model_class,
        model_name=model_name,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

