#!/usr/bin/env -S uv run
"""Quick test of model predictions vs expected values."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from tiny_icf.model import UniversalICF
from tiny_icf.data import load_frequency_list, compute_normalized_icf


def test_model(model_path: Path, data_path: Path):
    """Test model predictions against expected ICF values."""
    # Load expected ICF values
    word_counts, total_tokens = load_frequency_list(data_path)
    word_icf = compute_normalized_icf(word_counts, total_tokens)
    
    # Load model
    model = UniversalICF()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Test words
    test_cases = [
        ('the', 'very common'),
        ('you', 'very common'),
        ('apple', 'moderate'),
        ('xylophone', 'rare'),
        ('computer', 'moderate'),
        ('qzxbjk', 'impossible'),
    ]
    
    print(f"{'Word':<15} {'Predicted':<12} {'Expected':<12} {'Error':<12} {'Status':<20}")
    print("=" * 80)
    
    total_error = 0
    for word, desc in test_cases:
        word_lower = word.lower()
        
        # Get expected
        if word_lower in word_icf:
            expected = word_icf[word_lower]
        else:
            expected = None
        
        # Get prediction
        byte_seq = word_lower.encode('utf-8')[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            predicted = model(byte_tensor).item()
        
        if expected is not None:
            error = abs(predicted - expected)
            total_error += error
            status = "✓" if error < 0.1 else "✗"
            print(f"{word:<15} {predicted:<12.4f} {expected:<12.4f} {error:<12.4f} {status} {desc}")
        else:
            print(f"{word:<15} {predicted:<12.4f} {'N/A':<12} {'N/A':<12} ? {desc}")
    
    if total_error > 0:
        avg_error = total_error / len([w for w, _ in test_cases if w.lower() in word_icf])
        print("=" * 80)
        print(f"Average error: {avg_error:.4f}")


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "models" / "model_local_v2.pt"
    data_path = Path(__file__).parent.parent / "data" / "combined_frequencies.csv"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    test_model(model_path, data_path)

