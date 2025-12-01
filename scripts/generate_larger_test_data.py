"""Generate a larger test frequency list for better training."""

import csv
import random
import sys
from pathlib import Path

# Extended word list with more diversity
WORDS = [
    # Very common (stopwords)
    ("the", 1000000), ("be", 800000), ("to", 700000), ("of", 600000), ("and", 500000),
    ("a", 450000), ("in", 400000), ("that", 350000), ("have", 300000), ("i", 280000),
    ("it", 260000), ("for", 240000), ("not", 220000), ("on", 200000), ("with", 180000),
    ("he", 170000), ("as", 160000), ("you", 150000), ("do", 140000), ("at", 130000),
    
    # Common words
    ("this", 120000), ("but", 110000), ("his", 100000), ("by", 95000), ("from", 90000),
    ("they", 85000), ("we", 80000), ("say", 75000), ("her", 70000), ("she", 65000),
    ("or", 60000), ("an", 55000), ("will", 50000), ("my", 45000), ("one", 40000),
    ("all", 38000), ("would", 35000), ("there", 32000), ("their", 30000), ("what", 28000),
    
    # Medium frequency
    ("so", 26000), ("up", 24000), ("out", 22000), ("if", 20000), ("about", 18000),
    ("who", 16000), ("get", 15000), ("which", 14000), ("go", 13000), ("me", 12000),
    ("when", 11000), ("make", 10000), ("can", 9000), ("like", 8000), ("time", 7000),
    ("no", 6500), ("just", 6000), ("him", 5500), ("know", 5000), ("take", 4500),
    
    # Less common
    ("people", 4000), ("into", 3800), ("year", 3500), ("your", 3200), ("good", 3000),
    ("some", 2800), ("could", 2600), ("them", 2400), ("see", 2200), ("other", 2000),
    ("than", 1800), ("then", 1600), ("now", 1500), ("look", 1400), ("only", 1300),
    ("come", 1200), ("its", 1100), ("over", 1000), ("think", 900), ("also", 800),
    
    # Rare but valid words
    ("xylophone", 50), ("zephyr", 40), ("quixotic", 35), ("jubilant", 30),
    ("serendipity", 25), ("ephemeral", 20), ("luminous", 18), ("resilient", 15),
    ("mellifluous", 12), ("perspicacious", 10),
    
    # Very rare / technical
    ("algorithm", 500), ("quantum", 300), ("neural", 250), ("synthetic", 200),
    ("paradigm", 150), ("heuristic", 100), ("asymptotic", 80), ("polymorphic", 60),
    
    # Non-existent but plausible
    ("flimjam", 3), ("glimmerous", 2), ("quibblewock", 1),
    
    # Impossible structures
    ("qzxbjk", 1), ("xkjqpz", 1), ("zqkjxp", 1),
]

# Add more common words with variations
common_base = [
    "apple", "book", "car", "dog", "eye", "fish", "game", "house", "ice", "jump",
    "king", "love", "moon", "name", "open", "play", "queen", "rain", "star", "tree",
    "under", "very", "water", "xray", "yellow", "zebra"
]

for word in common_base:
    count = random.randint(500, 5000)
    WORDS.append((word, count))

# Add morphological variations
morphological = [
    ("running", 2000), ("runner", 500), ("runs", 3000),
    ("friend", 4000), ("friendly", 1500), ("friendship", 800), ("unfriendly", 200),
    ("happy", 5000), ("happiness", 2000), ("unhappy", 800),
    ("compute", 1000), ("computer", 5000), ("computation", 600), ("computational", 200),
    ("teach", 3000), ("teacher", 4000), ("teaching", 2500), ("teachable", 300),
]

WORDS.extend(morphological)

# Sort by frequency (descending)
WORDS.sort(key=lambda x: x[1], reverse=True)


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("larger_test_frequencies.csv")
    
    with open(output_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for word, count in WORDS:
            writer.writerow([word, count])
    
    total_tokens = sum(count for _, count in WORDS)
    print(f"Generated larger test frequency list: {output_path}")
    print(f"Words: {len(WORDS)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Frequency range: {WORDS[-1][1]:,} to {WORDS[0][1]:,}")


if __name__ == "__main__":
    main()

