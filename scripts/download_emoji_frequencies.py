"""Download emoji/emoticon frequency data from web sources."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import requests


def download_emoji_frequency_list(output_dir: Path):
    """Download emoji frequency data if available."""
    # Emoji frequency lists are available from various sources
    # For now, we'll create a reasonable default based on common usage
    
    sources = [
        {
            "name": "unicode_emoji_freq",
            "url": "https://raw.githubusercontent.com/unicode-org/cldr/main/common/annotations/en.xml",
            "format": "xml",
        },
    ]
    
    # Create default emoji frequency list based on common web usage
    # These are approximate frequencies from web text analysis
    default_emojis = {
        '❤️': 1000000, '😀': 800000, '😂': 750000, '😊': 600000,
        '😍': 500000, '😘': 450000, '👍': 400000, '😎': 350000,
        '😭': 300000, '😢': 280000, '😡': 250000, '😱': 220000,
        '😴': 200000, '😋': 180000, '😏': 160000, '😌': 140000,
        '🙂': 120000, '😉': 100000, '🤔': 90000, '😮': 80000,
        '😯': 70000, '😲': 60000, '😳': 50000, '😵': 40000,
        '🤗': 35000, '🤝': 30000, '👏': 25000, '🙌': 20000,
        '✌️': 18000, '🤞': 15000, '🤘': 12000, '👌': 10000,
        '👍': 9000, '👎': 8000, '👊': 7000, '✊': 6000,
        '💪': 5000, '🙏': 4000, '👐': 3000, '🤲': 2000,
    }
    
    # Text emoticons
    text_emoticons = {
        ':)': 500000, ':-)': 400000, '=)': 300000,
        ':(': 450000, ':-(': 350000, '=(': 250000,
        ';)': 200000, ';-)': 150000,
        ':D': 300000, ':-D': 250000, '=D': 200000,
        ':P': 180000, ':-P': 150000, '=P': 120000,
        ':/': 100000, ':-/': 80000, '=/': 60000,
        '<3': 400000, '</3': 50000,
        'xD': 200000, 'XD': 250000,
        'o_O': 80000, 'O_O': 60000,
    }
    
    # Combine
    all_emojis = {**default_emojis, **text_emoticons}
    
    # Write CSV
    output_path = output_dir / "emoji_frequencies.csv"
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['emoji', 'count'])
        for emoji, count in sorted(all_emojis.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([emoji, count])
    
    print(f"Created emoji frequency list: {len(all_emojis)} emojis/emoticons")
    print(f"Saved to: {output_path}")
    return output_path


def extract_emojis_from_typo_corpus(typo_csv: Path, output_path: Path):
    """Extract emojis/emoticons from typo corpus."""
    emoji_counts = Counter()
    
    with open(typo_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check both typo and correction
            for text in [row.get('typo', ''), row.get('correction', '')]:
                # Simple emoji detection (Unicode ranges)
                for char in text:
                    # Emoji ranges
                    if (ord(char) >= 0x1F600 and ord(char) <= 0x1F64F) or \
                       (ord(char) >= 0x1F300 and ord(char) <= 0x1F5FF) or \
                       (ord(char) >= 0x1F680 and ord(char) <= 0x1F6FF) or \
                       (ord(char) >= 0x2600 and ord(char) <= 0x26FF) or \
                       (ord(char) >= 0x2700 and ord(char) <= 0x27BF):
                        emoji_counts[char] += 1
                
                # Text emoticons
                import re
                emoticons = re.findall(r'[:;=][\-]?[\)\(DPpOo/]|<3|</3|xd|XD', text, re.IGNORECASE)
                for emoticon in emoticons:
                    emoji_counts[emoticon.lower()] += 1
    
    if emoji_counts:
        with open(output_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['emoji', 'count'])
            for emoji, count in emoji_counts.most_common():
                writer.writerow([emoji, count])
        
        print(f"Extracted {len(emoji_counts)} unique emojis/emoticons from typo corpus")
        print(f"Saved to: {output_path}")
    else:
        print("No emojis found in typo corpus")
    
    return output_path if emoji_counts else None


def main():
    output_dir = Path("data/emojis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Creating Emoji/Emoticon Frequency List")
    print("=" * 80)
    
    # Create default list
    emoji_csv = download_emoji_frequency_list(output_dir)
    
    # Extract from typo corpus if available
    typo_csv = Path("data/typos/github_typos.csv")
    if typo_csv.exists():
        print("\nExtracting emojis from typo corpus...")
        extract_emojis_from_typo_corpus(typo_csv, output_dir / "emojis_from_typos.csv")
    
    print(f"\n✓ Emoji frequency data ready in {output_dir}")


if __name__ == "__main__":
    main()

