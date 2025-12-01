"""Find real typos from Common Crawl by detecting low-frequency words
that are close to high-frequency words (typo → correction pattern).

This is better than mapping - we find REAL typo patterns with frequencies.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from Levenshtein import distance as levenshtein_distance


def find_typo_candidates(
    frequency_list: Dict[str, int],
    max_edit_distance: int = 2,
    min_correction_freq: int = 100,
    max_typo_freq: int = 10,
) -> List[Tuple[str, str, int, int]]:
    """
    Find typo candidates: low-frequency words close to high-frequency words.
    
    Returns:
        List of (typo, correction, typo_count, correction_count) tuples
    """
    # Sort by frequency
    sorted_words = sorted(frequency_list.items(), key=lambda x: x[1], reverse=True)
    
    # High-frequency words (likely corrections)
    high_freq_words = {
        word: count
        for word, count in sorted_words
        if count >= min_correction_freq
    }
    
    # Low-frequency words (potential typos)
    low_freq_words = {
        word: count
        for word, count in sorted_words
        if count <= max_typo_freq and count > 0
    }
    
    typo_pairs = []
    
    print(f"Searching for typo patterns...")
    print(f"  High-freq words (corrections): {len(high_freq_words):,}")
    print(f"  Low-freq words (potential typos): {len(low_freq_words):,}")
    
    # For each low-freq word, find nearest high-freq word
    for typo, typo_count in low_freq_words.items():
        best_match = None
        best_distance = max_edit_distance + 1
        
        for correction, correction_count in high_freq_words.items():
            # Skip if same length and very different
            if abs(len(typo) - len(correction)) > 2:
                continue
            
            dist = levenshtein_distance(typo, correction)
            if dist <= max_edit_distance and dist < best_distance:
                best_distance = dist
                best_match = correction
        
        if best_match:
            typo_pairs.append((
                typo,
                best_match,
                typo_count,
                frequency_list[best_match],
            ))
    
    # Sort by edit distance (closer = more likely typo)
    typo_pairs.sort(key=lambda x: levenshtein_distance(x[0], x[1]))
    
    print(f"  Found {len(typo_pairs):,} potential typo-correction pairs")
    
    return typo_pairs


def filter_high_confidence_typos(
    typo_pairs: List[Tuple[str, str, int, int]],
    min_freq_ratio: float = 10.0,
) -> List[Tuple[str, str, int, int]]:
    """
    Filter to high-confidence typos:
    - Correction frequency >> typo frequency
    - Edit distance <= 2
    - Typo looks like a real typo (not just rare word)
    """
    filtered = []
    
    for typo, correction, typo_count, correction_count in typo_pairs:
        # Correction should be much more frequent
        if correction_count / typo_count < min_freq_ratio:
            continue
        
        # Additional heuristics
        # Skip if typo is just a prefix/suffix of correction
        if typo in correction or correction in typo:
            # Might be valid (e.g., "run" vs "running")
            # But if frequency ratio is huge, likely typo
            if correction_count / typo_count > 100:
                filtered.append((typo, correction, typo_count, correction_count))
            continue
        
        filtered.append((typo, correction, typo_count, correction_count))
    
    print(f"  Filtered to {len(filtered):,} high-confidence typos")
    return filtered


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find real typos from frequency list (better than mapping)"
    )
    parser.add_argument(
        "--frequency-list",
        type=Path,
        required=True,
        help="Frequency list CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/typos/real_typos_from_freq.csv"),
        help="Output typo-correction pairs",
    )
    parser.add_argument(
        "--max-edit-distance",
        type=int,
        default=2,
        help="Maximum edit distance for typo detection",
    )
    parser.add_argument(
        "--min-correction-freq",
        type=int,
        default=100,
        help="Minimum frequency for correction word",
    )
    parser.add_argument(
        "--max-typo-freq",
        type=int,
        default=10,
        help="Maximum frequency for typo word",
    )
    
    args = parser.parse_args()
    
    # Load frequency list
    print("Loading frequency list...")
    frequency_list = {}
    with open(args.frequency_list, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                word = row[0].strip().lower()
                count = int(row[1])
                frequency_list[word] = count
    
    print(f"Loaded {len(frequency_list):,} words")
    
    # Find typo candidates
    typo_pairs = find_typo_candidates(
        frequency_list,
        max_edit_distance=args.max_edit_distance,
        min_correction_freq=args.min_correction_freq,
        max_typo_freq=args.max_typo_freq,
    )
    
    # Filter to high-confidence
    filtered = filter_high_confidence_typos(typo_pairs)
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['typo', 'correction', 'typo_count', 'correction_count'])
        for typo, correction, typo_count, correction_count in filtered:
            writer.writerow([typo, correction, typo_count, correction_count])
    
    print(f"\n✓ Saved {len(filtered):,} real typo pairs to {args.output}")
    print("\nThese are REAL typos found in the frequency data,")
    print("not synthetic mappings. Use for augmentation with correction's frequency.")


if __name__ == "__main__":
    try:
        import Levenshtein
    except ImportError:
        print("Error: python-Levenshtein not installed")
        print("Install with: pip install python-Levenshtein")
        sys.exit(1)
    
    main()

