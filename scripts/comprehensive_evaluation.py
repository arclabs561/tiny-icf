# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
#   "scipy>=1.10.0",
# ]
# ///
"""Comprehensive evaluation of trained models using multiple metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Optional

from tiny_icf.model import UniversalICF
from tiny_icf.model_residual import ResidualICF
from tiny_icf.data import WordICFDataset, load_frequency_list, compute_normalized_icf, stratified_sample
from tiny_icf.eval import evaluate_jabberwocky, compute_metrics
from tiny_icf.eval_rbo import rbo

def evaluate_model_comprehensive(
    model: torch.nn.Module,
    dataset: WordICFDataset,
    device: torch.device,
    model_name: str = "model"
) -> Dict[str, float]:
    """Comprehensive evaluation with multiple metrics."""
    model.eval()
    
    predictions = []
    targets = []
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                words = batch['bytes'].to(device)
                icf = batch['icf'].to(device)
            else:
                words, icf = batch
                words = words.to(device)
                icf = icf.to(device)
            
            pred = model(words).squeeze()
            predictions.append(pred.cpu().numpy())
            targets.append(icf.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Basic metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Correlation metrics
    if np.std(predictions) > 0 and np.std(targets) > 0:
        spearman, _ = spearmanr(predictions, targets)
        pearson, _ = pearsonr(predictions, targets)
        if np.isnan(spearman):
            spearman = 0.0
        if np.isnan(pearson):
            pearson = 0.0
    else:
        spearman = 0.0
        pearson = 0.0
    
    # Ranking metrics
    # Sort by predictions and targets, compute RBO
    pred_rank = np.argsort(predictions)
    target_rank = np.argsort(targets)
    
    # RBO (Rank-Biased Overlap) - position-biased ranking metric
    rbo_score = rbo(
        list(pred_rank),
        list(target_rank),
        p=0.9  # High p = emphasize top ranks
    )
    
    # Percentile accuracy: How well does model rank common vs rare?
    common_threshold = np.percentile(targets, 25)  # Bottom 25% = common
    rare_threshold = np.percentile(targets, 75)    # Top 25% = rare
    
    common_mask = targets < common_threshold
    rare_mask = targets > rare_threshold
    
    if common_mask.sum() > 0 and rare_mask.sum() > 0:
        common_pred_mean = predictions[common_mask].mean()
        rare_pred_mean = predictions[rare_mask].mean()
        separation = rare_pred_mean - common_pred_mean
    else:
        separation = 0.0
    
    return {
        'model': model_name,
        'mae': float(mae),
        'rmse': float(rmse),
        'spearman': float(spearman),
        'pearson': float(pearson),
        'rbo': float(rbo_score),
        'separation': float(separation),
        'mean_pred': float(predictions.mean()),
        'mean_target': float(targets.mean()),
        'std_pred': float(predictions.std()),
        'std_target': float(targets.std()),
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    word_counts, total_tokens = load_frequency_list(Path('data/word_frequency.csv'))
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    samples = stratified_sample(word_icf, word_counts=word_counts, use_token_frequency=False)
    split_idx = int(len(samples) * 0.8)
    test_samples = samples[split_idx:]
    test_dataset = WordICFDataset(test_samples, max_length=20, augment_prob=0.0)
    
    models_to_evaluate = [
        ('temporal_amoo', Path('models/model_temporal_amoo.pt'), UniversalICF),
        ('reduced_capacity', Path('models/model_reduced_capacity.pt'), lambda: UniversalICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4)),
        ('batchnorm', Path('models/model_batchnorm.pt'), lambda: UniversalICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4)),  # BatchNorm is now built-in
        ('residual', Path('models/model_residual.pt'), lambda: ResidualICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4)),
        ('aggressive_reg', Path('models/model_aggressive_reg.pt'), lambda: UniversalICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.5)),
    ]
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    print()
    
    results = []
    
    for name, model_path, model_class in models_to_evaluate:
        if not model_path.exists():
            print(f"{name}: Model not found, skipping...")
            continue
        
        try:
            if callable(model_class):
                model = model_class().to(device)
            else:
                model = model_class().to(device)
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            # Try strict loading first, fall back to non-strict if architecture mismatch
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except RuntimeError as e:
                if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e) or "size mismatch" in str(e):
                    print(f"  Warning: Architecture mismatch, loading with strict=False")
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    raise
            
            metrics = evaluate_model_comprehensive(model, test_dataset, device, name)
            results.append(metrics)
            
            print(f"{name.upper()}:")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  Spearman: {metrics['spearman']:.4f}")
            print(f"  Pearson: {metrics['pearson']:.4f}")
            print(f"  RBO: {metrics['rbo']:.4f}")
            print(f"  Separation: {metrics['separation']:.4f}")
            print()
            
        except Exception as e:
            print(f"{name}: Error - {e}")
            continue
    
    # Summary table
    if results:
        print("=" * 80)
        print("SUMMARY COMPARISON")
        print("=" * 80)
        print(f"{'Model':<20} {'Spearman':<12} {'RBO':<10} {'MAE':<10} {'Separation':<12}")
        print("-" * 80)
        for r in sorted(results, key=lambda x: x['spearman'], reverse=True):
            print(f"{r['model']:<20} {r['spearman']:<12.4f} {r['rbo']:<10.4f} {r['mae']:<10.4f} {r['separation']:<12.4f}")
        
        # Save results
        results_path = Path('models/comprehensive_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {results_path}")

if __name__ == "__main__":
    main()

