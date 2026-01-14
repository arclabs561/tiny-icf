# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""
Evaluate model robustness to adversarial examples, OOD words, and noise.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tiny_icf.eval_robustness import compute_robustness_metrics
from tiny_icf.data import WordICFDataset, load_frequency_list, compute_normalized_icf


def create_model_predictor(model, device, max_length=20):
    """Create a function that predicts ICF for a word."""
    def predict(word: str) -> float:
        # Convert word to bytes
        byte_seq = word.encode("utf-8")[:max_length]
        padded = byte_seq + bytes(max_length - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            icf = model(byte_tensor).item()
        
        return float(icf)
    
    return predict


def generate_ood_words() -> List[str]:
    """Generate out-of-distribution words for testing."""
    return [
        # Gibberish
        "qzxbjk", "flimjam", "xylophonic",
        # Code-like
        "var123", "func_xyz", "classABC",
        # Foreign characters (if model handles them)
        "café", "naïve", "résumé",
        # Very long words
        "supercalifragilisticexpialidocious",
        "pneumonoultramicroscopicsilicovolcanoconiosis",
        # Numbers and symbols
        "12345", "abc123", "test@123",
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    parser.add_argument("--output", type=Path, default=Path("robustness_results.json"), help="Output JSON path")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--n-test-words", type=int, default=100, help="Number of test words")
    parser.add_argument("--include-ood", action="store_true", help="Include OOD testing")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data}...")
    word_counts, total_tokens = load_frequency_list(args.data)
    icf_scores = compute_normalized_icf(word_counts, total_tokens)
    
    # Select test words
    all_words = list(icf_scores.keys())
    test_words = np.random.choice(all_words, size=min(args.n_test_words, len(all_words)), replace=False).tolist()
    
    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # Try to infer model class
    if 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
        if model_type == 'ResidualICF':
            from tiny_icf.model_residual import ResidualICF
            model = ResidualICF().to(device)
        elif model_type == 'NanoICF':
            from tiny_icf.nano_model import NanoICF
            model = NanoICF().to(device)
        else:
            from tiny_icf.model import UniversalICF
            model = UniversalICF().to(device)
    else:
        from tiny_icf.model import UniversalICF
        model = UniversalICF().to(device)
    
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()
    
    # Create predictor function
    predictor = create_model_predictor(model, device)
    
    # Generate OOD words if requested
    ood_words = None
    if args.include_ood:
        ood_words = generate_ood_words()
        print(f"Generated {len(ood_words)} OOD words")
    
    # Compute robustness metrics
    print("Computing robustness metrics...")
    robustness_results = compute_robustness_metrics(
        predictor,
        test_words,
        ood_words=ood_words,
    )
    
    # Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(robustness_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS EVALUATION RESULTS")
    print("=" * 70)
    
    if 'adversarial' in robustness_results:
        adv = robustness_results['adversarial']
        print(f"\nAdversarial Robustness:")
        print(f"  Mean perturbation error: {adv.get('mean_perturbation_error', 0.0):.4f}")
        print(f"  Max perturbation error: {adv.get('max_perturbation_error', 0.0):.4f}")
        print(f"  Robustness score: {adv.get('robustness_score', 0.0):.4f}")
    
    if 'noise' in robustness_results:
        print(f"\nNoise Robustness:")
        for noise_level, metrics in robustness_results['noise'].items():
            print(f"  {noise_level}:")
            print(f"    Mean error: {metrics.get('mean_error', 0.0):.4f}")
            print(f"    Robustness score: {metrics.get('robustness_score', 0.0):.4f}")
    
    if 'ood' in robustness_results:
        ood = robustness_results['ood']
        print(f"\nOOD Robustness:")
        print(f"  Mean OOD ICF: {ood.get('mean_ood_icf', 0.0):.4f}")
        print(f"  OOD detection rate: {ood.get('ood_detection_rate', 0.0):.4f}")
    
    if 'overall_robustness' in robustness_results:
        print(f"\nOverall Robustness Score: {robustness_results['overall_robustness']:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

