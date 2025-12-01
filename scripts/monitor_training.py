#!/usr/bin/env python3
"""Monitor training progress and test predictions periodically."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from tiny_icf.model import UniversalICF


def test_predictions(model_path: Path):
    """Test model predictions on key words."""
    if not model_path.exists():
        return None
    
    try:
        model = UniversalICF()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        test_words = [
            ('the', 0.17, 'very common'),
            ('apple', 0.52, 'moderate'),
            ('xylophone', 0.75, 'rare'),
            ('qzxbjk', 0.99, 'impossible'),
        ]
        
        results = []
        for word, expected, desc in test_words:
            byte_seq = word.encode('utf-8')[:20]
            padded = byte_seq + bytes(20 - len(byte_seq))
            byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                score = model(byte_tensor).item()
            error = abs(score - expected)
            results.append((word, score, expected, error, desc))
        
        return results
    except Exception as e:
        return f"Error: {e}"


def main():
    model_path = Path(__file__).parent.parent / "models" / "model_local_v2.pt"
    log_path = Path(__file__).parent.parent / "training_v2.log"
    
    print("Monitoring training...")
    print("=" * 80)
    
    last_size = 0
    while True:
        # Check log file
        if log_path.exists():
            current_size = log_path.stat().st_size
            if current_size > last_size:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    # Show last 10 lines
                    print("\n" + "=" * 80)
                    print(f"Latest log output (showing last 10 lines):")
                    print("=" * 80)
                    for line in lines[-10:]:
                        print(line.rstrip())
                    last_size = current_size
        
        # Test predictions if model exists
        results = test_predictions(model_path)
        if results:
            print("\n" + "=" * 80)
            print("Current Model Predictions:")
            print("=" * 80)
            print(f"{'Word':<15} {'Predicted':<12} {'Expected':<12} {'Error':<12} {'Desc':<20}")
            print("-" * 80)
            for word, pred, exp, err, desc in results:
                print(f"{word:<15} {pred:<12.4f} {exp:<12.4f} {err:<12.4f} {desc:<20}")
        
        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

