"""Generate a small test frequency list for development."""

import csv
import sys
from pathlib import Path

# Top 1000 most common English words with synthetic frequencies
# Frequencies follow approximate Zipfian distribution
COMMON_WORDS = [
    ("the", 1000000),
    ("be", 800000),
    ("to", 700000),
    ("of", 600000),
    ("and", 500000),
    ("a", 450000),
    ("in", 400000),
    ("that", 350000),
    ("have", 300000),
    ("i", 280000),
    ("it", 260000),
    ("for", 240000),
    ("not", 220000),
    ("on", 200000),
    ("with", 180000),
    ("he", 170000),
    ("as", 160000),
    ("you", 150000),
    ("do", 140000),
    ("at", 130000),
    ("this", 120000),
    ("but", 110000),
    ("his", 100000),
    ("by", 95000),
    ("from", 90000),
    ("they", 85000),
    ("we", 80000),
    ("say", 75000),
    ("her", 70000),
    ("she", 65000),
    ("or", 60000),
    ("an", 55000),
    ("will", 50000),
    ("my", 45000),
    ("one", 40000),
    ("all", 38000),
    ("would", 35000),
    ("there", 32000),
    ("their", 30000),
    ("what", 28000),
    ("so", 26000),
    ("up", 24000),
    ("out", 22000),
    ("if", 20000),
    ("about", 18000),
    ("who", 16000),
    ("get", 15000),
    ("which", 14000),
    ("go", 13000),
    ("me", 12000),
    ("when", 11000),
    ("make", 10000),
    ("can", 9000),
    ("like", 8000),
    ("time", 7000),
    ("no", 6500),
    ("just", 6000),
    ("him", 5500),
    ("know", 5000),
    ("take", 4500),
    ("people", 4000),
    ("into", 3800),
    ("year", 3500),
    ("your", 3200),
    ("good", 3000),
    ("some", 2800),
    ("could", 2600),
    ("them", 2400),
    ("see", 2200),
    ("other", 2000),
    ("than", 1800),
    ("then", 1600),
    ("now", 1500),
    ("look", 1400),
    ("only", 1300),
    ("come", 1200),
    ("its", 1100),
    ("over", 1000),
    ("think", 900),
    ("also", 800),
    ("back", 750),
    ("after", 700),
    ("use", 650),
    ("two", 600),
    ("how", 550),
    ("our", 500),
    ("work", 450),
    ("first", 400),
    ("well", 380),
    ("way", 350),
    ("even", 320),
    ("new", 300),
    ("want", 280),
    ("because", 260),
    ("any", 240),
    ("these", 220),
    ("give", 200),
    ("day", 180),
    ("most", 160),
    ("us", 150),
    ("xylophone", 5),  # Rare word for testing
    ("flimjam", 3),  # Non-existent but plausible
    ("qzxbjk", 1),  # Impossible structure
]


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_frequencies.csv")
    
    with open(output_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for word, count in COMMON_WORDS:
            writer.writerow([word, count])
    
    total_tokens = sum(count for _, count in COMMON_WORDS)
    print(f"Generated test frequency list: {output_path}")
    print(f"Words: {len(COMMON_WORDS)}")
    print(f"Total tokens: {total_tokens:,}")


if __name__ == "__main__":
    main()

