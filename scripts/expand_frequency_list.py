"""Expand frequency list from Common Crawl or other large corpora.

Instead of mapping OOV words to nearest neighbors (problematic),
we extract words that have real frequency counts.
"""

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Set

import requests


def download_common_crawl_word_counts(output_dir: Path):
    """Download Common Crawl word frequency lists if available."""
    # Common Crawl word counts are available from various sources
    # For now, we'll use a placeholder - user can provide their own
    
    sources = [
        # Add Common Crawl word count sources here
        # Example: "https://commoncrawl.org/word-frequencies"
    ]
    
    print("Common Crawl word counts need to be downloaded separately.")
    print("Options:")
    print("1. Download from Common Crawl website")
    print("2. Process Common Crawl yourself (large dataset)")
    print("3. Use pre-processed word counts from research groups")
    
    return None


def merge_frequency_lists(
    existing_path: Path,
    new_path: Path,
    output_path: Path,
    min_count: int = 5,
):
    """
    Merge two frequency lists, combining counts for duplicate words.
    
    This is better than mapping - we use real frequencies.
    """
    # Load existing
    existing_counts = {}
    with open(existing_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                word = row[0].strip().lower()
                count = int(row[1])
                existing_counts[word] = count
    
    # Load new
    new_counts = {}
    with open(new_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                word = row[0].strip().lower()
                count = int(row[1])
                new_counts[word] = count
    
    # Merge (sum counts for duplicates)
    merged = Counter(existing_counts)
    merged.update(new_counts)
    
    # Filter by min_count
    filtered = {word: count for word, count in merged.items() if count >= min_count}
    
    # Write merged
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for word, count in sorted(filtered.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, count])
    
    print(f"Merged frequency lists:")
    print(f"  Existing: {len(existing_counts):,} words")
    print(f"  New: {len(new_counts):,} words")
    print(f"  Merged: {len(filtered):,} words (min_count={min_count})")
    print(f"  Saved to: {output_path}")
    
    return output_path


def extract_high_confidence_words(
    frequency_list: Path,
    min_count: int = 10,
    output_path: Path | None = None,
) -> Dict[str, int]:
    """
    Extract words with high-confidence frequency counts.
    
    Instead of mapping OOV words, we only use words with reliable counts.
    """
    word_counts = {}
    
    with open(frequency_list, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                word = row[0].strip().lower()
                count = int(row[1])
                if count >= min_count:
                    word_counts[word] = count
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([word, count])
        print(f"Extracted {len(word_counts):,} high-confidence words (min_count={min_count})")
        print(f"Saved to: {output_path}")
    
    return word_counts


def find_oov_words_for_evaluation(
    common_crawl_words: Set[str],
    frequency_list_words: Set[str],
    output_path: Path,
):
    """
    Find OOV words from Common Crawl for evaluation (not training).
    
    These are words we don't have frequencies for - use for zero-shot testing.
    """
    oov_words = common_crawl_words - frequency_list_words
    
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for word in sorted(oov_words):
            writer.writerow([word])
    
    print(f"Found {len(oov_words):,} OOV words for evaluation")
    print(f"Saved to: {output_path}")
    print("\nUse these for zero-shot evaluation, NOT training.")
    print("Model should predict reasonable frequencies without labels.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Expand frequency list (better than nearest-word mapping)"
    )
    parser.add_argument(
        "--existing",
        type=Path,
        required=True,
        help="Existing frequency list CSV",
    )
    parser.add_argument(
        "--new",
        type=Path,
        help="New frequency list to merge",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output merged frequency list",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum count threshold",
    )
    
    args = parser.parse_args()
    
    if args.new:
        # Merge frequency lists
        merge_frequency_lists(
            args.existing,
            args.new,
            args.output,
            min_count=args.min_count,
        )
    else:
        # Just extract high-confidence words
        extract_high_confidence_words(
            args.existing,
            min_count=args.min_count,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()

