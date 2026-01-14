# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "scipy>=1.10.0",
# ]
# ///
"""Evaluate model robustness to various perturbations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from scipy.stats import spearmanr
from tiny_icf.model import UniversalICF
from tiny_icf.data import WordICFDataset, load_frequency_list, compute_normalized_icf

def add_noise(bytes_tensor: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """Add random noise to byte tensor."""
    noise = torch.randint(0, int(256 * noise_level), bytes_tensor.shape)
    noisy = torch.clamp(bytes_tensor + noise, 0, 255)
    return noisy

def test_robustness(model, device, test_words: list, icf_scores: dict):
    """Test model robustness to perturbations."""
    model.eval()
    
    results = {
        'original': [],
        'noise_0.05': [],
        'noise_0.1': [],
        'noise_0.2': [],
    }
    
    # Create a temporary dataset to get byte encoding
    temp_dataset = WordICFDataset([(w, 0.5) for w in test_words], max_length=20)
    
    with torch.no_grad():
        for i, word in enumerate(test_words):
            item = temp_dataset[i]
            original_bytes = (item['bytes'] if isinstance(item, dict) else item[0]).unsqueeze(0).to(device)
            original_pred = model(original_bytes).item()
            results['original'].append(original_pred)
            
            # Test with noise
            for noise_level in [0.05, 0.1, 0.2]:
                noisy_bytes = add_noise(original_bytes, noise_level)
                noisy_pred = model(noisy_bytes).item()
                results[f'noise_{noise_level}'].append(noisy_pred)
    
    # Calculate correlations
    original_preds = np.array(results['original'])
    target_icfs = np.array([icf_scores.get(w, 0.5) for w in test_words])
    
    print("=" * 80)
    print("ROBUSTNESS EVALUATION")
    print("=" * 80)
    print(f"\nTest words: {len(test_words)}")
    
    # Original performance
    if len(original_preds) > 1 and np.std(original_preds) > 0 and np.std(target_icfs) > 0:
        corr, _ = spearmanr(original_preds, target_icfs)
        mae = np.mean(np.abs(original_preds - target_icfs))
        print(f"\nOriginal Performance:")
        print(f"  Spearman: {corr:.4f}")
        print(f"  MAE: {mae:.4f}")
    
    # Robustness to noise
    print(f"\nRobustness to Noise:")
    for noise_level in [0.05, 0.1, 0.2]:
        noisy_preds = np.array(results[f'noise_{noise_level}'])
        pred_diff = np.abs(noisy_preds - original_preds)
        mean_diff = np.mean(pred_diff)
        max_diff = np.max(pred_diff)
        print(f"  Noise {noise_level:.2f}: Mean diff: {mean_diff:.4f}, Max diff: {max_diff:.4f}")
        
        # Correlation with targets
        if len(noisy_preds) > 1 and np.std(noisy_preds) > 0 and np.std(target_icfs) > 0:
            corr, _ = spearmanr(noisy_preds, target_icfs)
            print(f"    Spearman: {corr:.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model - use reduced capacity model to match current architecture
    from tiny_icf.model import UniversalICF
    model_path = Path('models/model_reduced_capacity.pt')
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Using untrained model for testing...")
        model = UniversalICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4).to(device)
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = UniversalICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    
    # Load test words
    word_counts, total_tokens = load_frequency_list(Path('data/word_frequency.csv'))
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    
    # Select diverse test words
    test_words = [
        'the', 'and', 'hello', 'world',
        'xylophone', 'quixotic', 'serendipity',
        'qzxbjk', 'asdfgh', 'test123',
    ]
    
    test_robustness(model, device, test_words, word_icf)

if __name__ == "__main__":
    main()

