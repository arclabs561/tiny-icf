#!/usr/bin/env -S uv run
"""Download and process Google Books Ngram data for training."""

import sys
from pathlib import Path
import subprocess
import csv
from collections import defaultdict

# Google Books Ngram download URLs
# Format: https://storage.googleapis.com/books/ngrams/books/20200217/eng/eng-1-ngrams_exports.html
NGRAM_BASE_URL = "https://storage.googleapis.com/books/ngrams/books/20200217/eng/"

# We want 1-grams (single words) from recent years
# Files are large, so we'll download specific ranges
NGRAM_FILES = [
    # Recent years (2000-2019) - most relevant for modern words
    "eng-1-ngrams_exports.html",  # Index page
    # Actual data files are like: eng-1-00000-of-00001.gz
]

def download_ngram_index():
    """Download the index page to see available files."""
    import urllib.request
    
    index_url = NGRAM_BASE_URL + "eng-1-ngrams_exports.html"
    output_path = Path(__file__).parent.parent / "data" / "ngrams" / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading ngram index from {index_url}...")
    try:
        urllib.request.urlretrieve(index_url, output_path)
        print(f"✓ Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        print("Note: Google Books Ngram data is very large.")
        print("Alternative: Use pre-processed frequency lists or Common Crawl data.")
        return None


def download_common_crawl_frequencies():
    """Alternative: Download Common Crawl word frequencies."""
    print("\nAlternative: Common Crawl word frequencies")
    print("=" * 70)
    print("Common Crawl provides word frequency data from recent web crawls.")
    print("Sources:")
    print("1. https://commoncrawl.org/ - Raw crawl data")
    print("2. https://github.com/facebookresearch/cc_net - Processed data")
    print("3. https://wortschatz.uni-leipzig.de/en/download - Pre-processed frequencies")
    print()
    print("For now, we'll use existing data and add modern words manually.")


def process_blog_frequencies():
    """Process blog/website word frequencies from recent sources."""
    print("\nBlog/Recent Text Frequencies:")
    print("=" * 70)
    print("Options:")
    print("1. Scrape recent blog posts (requires web scraping)")
    print("2. Use Reddit/Twitter word frequencies (social media)")
    print("3. Use Wikipedia word frequencies (encyclopedic)")
    print("4. Use news article frequencies (recent events)")
    print()
    print("For now, we'll focus on improving existing data.")


def main():
    print("Google Books Ngram Data Download")
    print("=" * 70)
    print()
    print("Note: Google Books Ngram data is very large (terabytes).")
    print("We'll download a small sample or use alternative sources.")
    print()
    
    # Try downloading index
    index_path = download_ngram_index()
    
    if index_path and index_path.exists():
        print(f"\n✓ Index downloaded. Review {index_path} for available files.")
        print("  Files are typically 1-5GB each, so download selectively.")
    else:
        print("\nUsing alternative approach:")
        download_common_crawl_frequencies()
        process_blog_frequencies()
    
    print("\n" + "=" * 70)
    print("Recommendation:")
    print("1. Use existing word_frequency.csv + word_frequency_modern.csv")
    print("2. Add more modern words manually (via add_modern_words.py)")
    print("3. For large-scale: Download Common Crawl processed frequencies")
    print("4. For recent: Scrape blog/news frequencies (future work)")


if __name__ == "__main__":
    main()

