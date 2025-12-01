#!/usr/bin/env python3
"""Demo: Optimal text reduction using ICF with real embeddings."""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from tiny_icf.model import UniversalICF
from tiny_icf.text_reduction_real import reduce_text_with_real_embeddings


def main():
    parser = argparse.ArgumentParser(description="Demo: ICF-guided text reduction with real embeddings")
    parser.add_argument("--model", type=Path, default=Path("models/model_local_v3.pt"), help="ICF model path")
    parser.add_argument("--text", type=str, default="the quick brown fox jumps over the lazy dog", help="Text to reduce")
    parser.add_argument("--ratio", type=float, default=0.5, help="Fraction of words to keep")
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "optimal"], help="Reduction method")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load ICF model
    icf_model = UniversalICF()
    if args.model.exists():
        icf_model.load_state_dict(torch.load(args.model, map_location=device))
        icf_model.eval()
        icf_model.to(device)
        print(f"✓ Loaded ICF model from {args.model}")
    else:
        print(f"⚠ Model not found: {args.model}")
        print("  Using untrained model for demo")
        icf_model.eval()
        icf_model.to(device)
    
    print("\n" + "=" * 70)
    print("ICF-Guided Text Reduction (with Real Embeddings)")
    print("=" * 70)
    print(f"\nOriginal text: {args.text}")
    print(f"Target: Keep {args.ratio*100:.0f}% of words")
    print(f"Method: {args.method}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Device: {device}")
    
    try:
        # Reduce
        reduced, regret, stats = reduce_text_with_real_embeddings(
            text=args.text,
            icf_model=icf_model,
            target_ratio=args.ratio,
            method=args.method,
            embedding_model=args.embedding_model,
            device=device,
        )
        
        print(f"\nReduced text: {reduced}")
        print(f"\nStatistics:")
        print(f"  Original length: {stats['original_length']} words")
        print(f"  Reduced length: {stats['reduced_length']} words")
        print(f"  Reduction: {stats['reduction_ratio']*100:.1f}%")
        if 'avg_icf_kept' in stats:
            print(f"  Avg ICF (kept): {stats['avg_icf_kept']:.4f}")
        if 'avg_icf_dropped' in stats:
            print(f"  Avg ICF (dropped): {stats['avg_icf_dropped']:.4f}")
        print(f"  Embedding regret: {stats['regret']:.4f}")
        if 'iterations' in stats:
            print(f"  Iterations: {stats['iterations']}")
        
        print("\n" + "=" * 70)
        print("Key Insight:")
        print("  By keeping high-ICF words (rare, informative) and dropping")
        print("  low-ICF words (common, less informative), we minimize")
        print("  embedding regret while preserving semantic content.")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use real embeddings, install sentence-transformers:")
        print("  pip install sentence-transformers")
        print("\nOr use the demo without real embeddings:")
        print("  python scripts/demo_text_reduction.py")


if __name__ == "__main__":
    main()

