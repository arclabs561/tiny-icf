#!/usr/bin/env python3
"""Add modern neologisms and portmanteaus to training data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.data import load_frequency_list, compute_normalized_icf
import csv

# Modern words with estimated frequencies (based on web usage)
# Format: (word, estimated_count, source)
MODERN_WORDS = [
    # Portmanteaus (not in current data)
    ('frenemy', 500, 'modern_portmanteau'),
    ('hangry', 800, 'modern_portmanteau'),
    ('webinar', 2000, 'modern_portmanteau'),
    ('spork', 200, 'modern_portmanteau'),
    ('chillax', 300, 'modern_portmanteau'),
    ('glamping', 400, 'modern_portmanteau'),
    ('mockumentary', 150, 'modern_portmanteau'),
    ('edutainment', 250, 'modern_portmanteau'),
    
    # Neologisms (not in current data)
    ('uberize', 100, 'tech_neologism'),
    ('cryptocurrency', 5000, 'tech_neologism'),
    ('blockchain', 3000, 'tech_neologism'),
    ('mansplain', 600, 'social_neologism'),
    ('hashtag', 8000, 'social_media'),
    ('emoji', 5000, 'social_media'),
    ('vlog', 1500, 'social_media'),
    ('podcast', 10000, 'media'),
    ('meme', 12000, 'internet'),
    ('viral', 8000, 'internet'),
    ('trending', 6000, 'social_media'),
    ('influencer', 4000, 'social_media'),
    ('unfriend', 2000, 'social_media'),
    ('retweet', 5000, 'social_media'),
    ('app', 50000, 'tech'),  # Very common now
    ('streaming', 15000, 'media'),
    ('algorithm', 20000, 'tech'),
    ('metadata', 5000, 'tech'),
    ('api', 30000, 'tech'),
]


def add_modern_words_to_frequency_list(
    input_path: Path,
    output_path: Path,
    modern_words: list[tuple[str, int, str]],
):
    """Add modern words to frequency list."""
    # Load existing data
    word_counts, total_tokens = load_frequency_list(input_path)
    
    print(f"Original data: {len(word_counts):,} words, {total_tokens:,} total tokens")
    
    # Add modern words
    added_count = 0
    updated_count = 0
    
    for word, count, source in modern_words:
        word_lower = word.lower()
        if word_lower in word_counts:
            # Update if new count is higher
            if count > word_counts[word_lower]:
                word_counts[word_lower] = count
                updated_count += 1
                print(f"  Updated: {word} ({source}) -> {count:,}")
        else:
            word_counts[word_lower] = count
            added_count += 1
            print(f"  Added: {word} ({source}) -> {count:,}")
    
    # Recompute ICF
    print(f"\nRecomputing ICF with new words...")
    word_icf = compute_normalized_icf(word_counts, total_tokens + sum(c for _, c, _ in modern_words))
    
    # Save updated frequency list
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
            writer.writerow([word, count])
    
    print(f"\n✓ Updated frequency list:")
    print(f"  Added: {added_count} words")
    print(f"  Updated: {updated_count} words")
    print(f"  Total: {len(word_counts):,} words")
    print(f"  Output: {output_path}")


def main():
    input_path = Path(__file__).parent.parent / "data" / "word_frequency.csv"
    output_path = Path(__file__).parent.parent / "data" / "word_frequency_modern.csv"
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    print("Adding modern neologisms and portmanteaus to training data...")
    print("=" * 70)
    
    add_modern_words_to_frequency_list(input_path, output_path, MODERN_WORDS)
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print(f"1. Review {output_path}")
    print("2. Retrain model with: python -m tiny_icf.train --data data/word_frequency_modern.csv")
    print("3. Test on portmanteaus/neologisms")


if __name__ == "__main__":
    main()

