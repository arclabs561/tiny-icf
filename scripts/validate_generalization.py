"""Validate model generalization on OOV test set."""

import sys
from pathlib import Path
import csv
import torch
import numpy as np
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tiny_icf.data import WordICFDataset
from tiny_icf.eval import compute_metrics


def load_oov_test_set(oov_path: Path):
    """Load OOV test set from CSV."""
    word_icf_pairs = []
    with open(oov_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['word'].strip()
            icf = float(row['icf_score'])
            word_icf_pairs.append((word, icf))
    return word_icf_pairs


def validate_generalization(
    model_path: Path,
    oov_test_path: Path,
    device: str = 'cpu',
    batch_size: int = 256,
):
    """
    Validate model generalization on OOV test set.
    
    Args:
        model_path: Path to trained model checkpoint
        oov_test_path: Path to OOV test set CSV
        device: Device to run on ('cpu' or 'cuda')
        batch_size: Batch size for evaluation
    
    Returns:
        Dict with metrics: spearman, mse, mae, etc.
    """
    print(f"Loading model from: {model_path}")
    
    # Load model
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        checkpoint = torch.load(model_path, map_location=device)
    else:
        device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model from checkpoint (PyTorch Lightning format)
    if 'state_dict' in checkpoint:
        # Lightning checkpoint
        from tiny_icf.model import UniversalICF
        model = UniversalICF()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Direct model checkpoint
        from tiny_icf.model import UniversalICF
        model = UniversalICF()
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    # Load OOV test set
    print(f"Loading OOV test set from: {oov_test_path}")
    oov_pairs = load_oov_test_set(oov_test_path)
    print(f"✅ Loaded {len(oov_pairs):,} OOV words")
    
    # Create dataset
    oov_dataset = WordICFDataset(oov_pairs, max_length=20, augment_prob=0.0)
    oov_loader = torch.utils.data.DataLoader(
        oov_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No multiprocessing for simplicity
    )
    
    # Evaluate
    print("\nEvaluating on OOV test set...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (byte_tensors, icf_targets) in enumerate(oov_loader):
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            predictions = model(byte_tensors)
            predictions = torch.clamp(predictions, 0.0, 1.0)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(icf_targets.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, targets)
    
    # Add Spearman correlation
    spearman_corr, spearman_p = spearmanr(predictions, targets)
    metrics['spearman_correlation'] = spearman_corr
    metrics['spearman_pvalue'] = spearman_p
    
    # Print results
    print("\n" + "=" * 70)
    print("Generalization Results (OOV Test Set)")
    print("=" * 70)
    print(f"Spearman Correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")
    print(f"MSE: {metrics.get('mse', 0):.6f}")
    print(f"MAE: {metrics.get('mae', 0):.6f}")
    print(f"RMSE: {metrics.get('rmse', 0):.6f}")
    print(f"R²: {metrics.get('r2', 0):.4f}")
    
    # Interpretation
    if spearman_corr > 0.5:
        print("\n✅ EXCELLENT generalization (Spearman > 0.5)")
    elif spearman_corr > 0.3:
        print("\n⚠️  MODERATE generalization (Spearman > 0.3)")
    elif spearman_corr > 0.1:
        print("\n⚠️  WEAK generalization (Spearman > 0.1)")
    else:
        print("\n❌ POOR generalization (Spearman < 0.1) - model may be overfitting")
    
    return metrics


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate model generalization on OOV test set')
    parser.add_argument('--model', type=Path, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--oov-test', type=Path, default=Path('data/oov_test_set.csv'),
                        help='Path to OOV test set CSV')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run on (default: cpu)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for evaluation (default: 256)')
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return 1
    
    if not args.oov_test.exists():
        print(f"❌ Error: OOV test set not found: {args.oov_test}")
        print(f"   Create it first with: python scripts/create_oov_test_set.py")
        return 1
    
    validate_generalization(
        model_path=args.model,
        oov_test_path=args.oov_test,
        device=args.device,
        batch_size=args.batch_size,
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

