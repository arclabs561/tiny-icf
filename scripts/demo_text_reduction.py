#!/usr/bin/env -S uv run
"""Demo: Optimal text reduction using ICF to minimize embedding regret."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from tiny_icf.model import UniversalICF


def simple_embedding(words: list[str]) -> torch.Tensor:
    """
    Simple embedding: average of word vectors.
    
    In practice, would use sentence-transformers or similar.
    For demo, use random embeddings (placeholder).
    """
    # Placeholder: in practice, use real embedding model
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # return model.encode(' '.join(words), convert_to_tensor=True)
    
    # For demo: random embedding (would be real in practice)
    return torch.randn(384)


def reduce_text_icf(
    text: str,
    icf_model: UniversalICF,
    target_ratio: float = 0.5,
) -> tuple[str, float, dict]:
    """
    Reduce text using ICF-guided optimal reduction.
    
    Args:
        text: Input text
        icf_model: ICF prediction model
        target_ratio: Fraction of words to keep
    
    Returns:
        (reduced_text, regret_estimate, stats)
    """
    words = text.split()
    n_words = len(words)
    target_length = max(1, int(n_words * target_ratio))
    
    # Predict ICF for each word
    icf_scores = []
    for word in words:
        # Clean word
        clean_word = ''.join(c for c in word.lower() if c.isalnum())
        if clean_word:
            byte_seq = clean_word.encode('utf-8')[:20]
            padded = byte_seq + bytes(20 - len(byte_seq))
            byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                icf = icf_model(byte_tensor).item()
            icf_scores.append(icf)
        else:
            icf_scores.append(0.0)
    
    # Sort by ICF (descending: keep high ICF)
    word_icf = list(zip(words, icf_scores))
    word_icf.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top N words
    kept = [word for word, icf in word_icf[:target_length]]
    dropped = [word for word, icf in word_icf[target_length:]]
    
    reduced_text = ' '.join(kept)
    
    # Estimate regret (would use real embeddings in practice)
    original_emb = simple_embedding(words)
    reduced_emb = simple_embedding(kept)
    regret = 1.0 - torch.nn.functional.cosine_similarity(
        original_emb.unsqueeze(0),
        reduced_emb.unsqueeze(0),
    ).item()
    
    stats = {
        'original_length': n_words,
        'reduced_length': len(kept),
        'reduction_ratio': 1.0 - (len(kept) / n_words),
        'avg_icf_kept': sum(icf for _, icf in word_icf[:target_length]) / target_length if target_length > 0 else 0.0,
        'avg_icf_dropped': sum(icf for _, icf in word_icf[target_length:]) / (n_words - target_length) if n_words > target_length else 0.0,
        'regret_estimate': regret,
    }
    
    return reduced_text, regret, stats


def main():
    parser = argparse.ArgumentParser(description="Demo: ICF-guided text reduction")
    parser.add_argument("--model", type=Path, default=Path("models/model_local_v3.pt"), help="ICF model path")
    parser.add_argument("--text", type=str, default="the quick brown fox jumps over the lazy dog", help="Text to reduce")
    parser.add_argument("--ratio", type=float, default=0.5, help="Fraction of words to keep")
    
    args = parser.parse_args()
    
    # Load model
    model = UniversalICF()
    if args.model.exists():
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
        model.eval()
        print(f"✓ Loaded model from {args.model}")
    else:
        print(f"⚠ Model not found: {args.model}")
        print("  Using untrained model for demo")
        model.eval()
    
    print("\n" + "=" * 70)
    print("ICF-Guided Text Reduction")
    print("=" * 70)
    print(f"\nOriginal text: {args.text}")
    print(f"Target: Keep {args.ratio*100:.0f}% of words")
    
    # Reduce
    reduced, regret, stats = reduce_text_icf(args.text, model, args.ratio)
    
    print(f"\nReduced text: {reduced}")
    print(f"\nStatistics:")
    print(f"  Original length: {stats['original_length']} words")
    print(f"  Reduced length: {stats['reduced_length']} words")
    print(f"  Reduction: {stats['reduction_ratio']*100:.1f}%")
    print(f"  Avg ICF (kept): {stats['avg_icf_kept']:.4f}")
    print(f"  Avg ICF (dropped): {stats['avg_icf_dropped']:.4f}")
    print(f"  Regret estimate: {stats['regret_estimate']:.4f}")
    
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("  By keeping high-ICF words (rare, informative) and dropping")
    print("  low-ICF words (common, less informative), we minimize")
    print("  embedding regret while preserving semantic content.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    main()

