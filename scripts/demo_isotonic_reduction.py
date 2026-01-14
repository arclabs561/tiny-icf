#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0.0",
#   "sentence-transformers>=2.2.0",
# ]
# ///
"""Demo: Isotonic regret text reduction.

Shows how regret increases as words are removed one at a time.
"""

import argparse
import json
from pathlib import Path

import torch

from tiny_icf.model import UniversalICF
from tiny_icf.text_reduction_isotonic import reduce_text_isotonic


def main():
    parser = argparse.ArgumentParser(description="Demo: Isotonic regret text reduction")
    parser.add_argument("--model", type=Path, default="models/model.pt", help="ICF model path")
    parser.add_argument("--text", type=str, default="the quick brown fox jumps over the lazy dog", help="Text to reduce")
    parser.add_argument("--target-ratio", type=float, default=0.5, help="Fraction of words to keep")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--no-isotonic", action="store_true", help="Don't enforce isotonic property")
    parser.add_argument("--verbose", action="store_true", help="Print progress at each step")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model}...")
    model = UniversalICF()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Reduce text
    print(f"\nOriginal text: {args.text}")
    print(f"Target ratio: {args.target_ratio} (keep {int(len(args.text.split()) * args.target_ratio)} words)")
    print(f"Enforce isotonic: {not args.no_isotonic}")
    print("\nReducing text...")
    
    reduced, regret, stats = reduce_text_isotonic(
        text=args.text,
        icf_model=model,
        target_ratio=args.target_ratio,
        embedding_model=args.embedding_model,
        device=device,
        enforce_isotonic=not args.no_isotonic,
        verbose=args.verbose,
    )
    
    print(f"\n✓ Reduction complete")
    print(f"\nReduced text: {reduced}")
    print(f"Final regret: {regret:.4f}")
    print(f"Words: {stats['reduced_length']}/{stats['original_length']} ({stats['reduction_ratio']:.1%} reduction)")
    print(f"Isotonic: {stats['is_isotonic']}")
    print(f"Steps: {stats['steps']}")
    
    # Show progression
    print("\nProgression:")
    print("Step | Words | Regret | Δ Regret | Word Removed | ICF")
    print("-" * 70)
    for p in stats['progression']:
        print(
            f"{p['step']:4d} | {p['words_remaining']:5d} | {p['regret']:6.4f} | "
            f"{p['regret_delta']:+8.4f} | {p['word_removed']:12s} | {p['icf_removed']:.3f}"
        )
    
    # Show regret curve
    print(f"\nRegret curve: {[f'{r:.4f}' for r in stats['regret_curve']]}")
    print(f"Max regret increase: {stats['max_regret_increase']:.4f}")
    print(f"Min regret increase: {stats['min_regret_increase']:.4f}")
    
    # Save if requested
    if args.output:
        output_data = {
            'original_text': args.text,
            'reduced_text': reduced,
            'final_regret': regret,
            'stats': stats,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()

