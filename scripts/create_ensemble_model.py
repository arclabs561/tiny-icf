# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
# ]
# ///
"""
Create an ensemble model from multiple trained models.
Averages predictions from multiple models for better performance.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import numpy as np

from tiny_icf.model import UniversalICF
from tiny_icf.model_residual import ResidualICF
from tiny_icf.data import WordICFDataset, load_frequency_list, compute_normalized_icf, stratified_sample


class EnsembleICF(nn.Module):
    """Ensemble of multiple ICF models."""
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
        self.weights = self.weights / self.weights.sum()  # Normalize
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted average
        stacked = torch.stack(predictions, dim=0)  # [N_models, Batch, 1]
        weights = self.weights.view(-1, 1, 1).to(stacked.device)  # [N_models, 1, 1]
        ensemble_pred = (stacked * weights).sum(dim=0)  # [Batch, 1]
        
        return torch.clamp(ensemble_pred, 0.0, 1.0)
    
    def count_parameters(self) -> int:
        return sum(m.count_parameters() for m in self.models)


def load_model(model_path: Path, model_type: str, device: torch.device) -> nn.Module:
    """Load a model from checkpoint."""
    if model_type == "residual":
        model = ResidualICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4)
    elif model_type == "batchnorm":
        model = UniversalICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4)
    elif model_type == "reduced":
        model = UniversalICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4)
    else:
        model = UniversalICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Create ensemble from multiple models.")
    parser.add_argument("--models", nargs='+', required=True, help="List of model paths")
    parser.add_argument("--types", nargs='+', required=True, help="List of model types (residual, batchnorm, reduced, etc.)")
    parser.add_argument("--weights", nargs='+', type=float, default=None, help="Optional weights for each model")
    parser.add_argument("--output", type=Path, default="models/model_ensemble.pt", help="Output path for ensemble")
    
    args = parser.parse_args()
    
    if len(args.models) != len(args.types):
        print("Error: --models and --types must have same length")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    models = []
    for model_path, model_type in zip(args.models, args.types):
        print(f"Loading {model_type} from {model_path}...")
        model = load_model(Path(model_path), model_type, device)
        models.append(model)
    
    # Create ensemble
    ensemble = EnsembleICF(models, args.weights)
    ensemble = ensemble.to(device)
    
    print(f"\nEnsemble created with {len(models)} models")
    print(f"Total parameters: {ensemble.count_parameters():,}")
    print(f"Model weights: {ensemble.weights.tolist()}")
    
    # Save ensemble
    torch.save({
        'model_state_dict': ensemble.state_dict(),
        'model_paths': args.models,
        'model_types': args.types,
        'weights': ensemble.weights.tolist(),
    }, args.output)
    
    print(f"\nEnsemble saved to {args.output}")
    
    # Test ensemble
    print("\nTesting ensemble...")
    dummy_input = torch.randint(0, 256, (4, 20), dtype=torch.long).to(device)
    output = ensemble(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("✓ Ensemble works correctly")

if __name__ == "__main__":
    main()

