# Data Preparation Guide

## Frequency List Sources

The model requires a frequency list in CSV format: `word,count` (one per line).

### Recommended Sources

1. **Google Web Trillion Word Corpus (1T)**
   - Download from: [Google Books Ngram](https://storage.googleapis.com/books/ngrams/books/datasetsv3.html)
   - Format: Already in tab-separated format, convert to CSV

2. **SymSpell Frequency Lists**
   - Available on GitHub: Various language frequency lists
   - Format: Usually `word count` (space-separated), convert to CSV

3. **Common Crawl Derived Lists**
   - Various research groups publish frequency lists
   - Ensure they include total token counts

### Example: Creating a Test Frequency List

For testing purposes, you can create a small frequency list:

```python
import csv

# Example: Top 1000 words with synthetic counts
words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i"]
counts = [1000000, 800000, 700000, 600000, 500000, 400000, 300000, 200000, 150000, 100000]

with open("test_frequencies.csv", "w") as f:
    writer = csv.writer(f)
    for word, count in zip(words, counts):
        writer.writerow([word, count])
```

### Data Format

The CSV should have:
- **Header row (optional)**: If present, will be skipped
- **Data rows**: `word,count` where:
  - `word`: Lowercase word (will be lowercased automatically)
  - `count`: Integer frequency count

Example:
```csv
the,1000000
be,800000
to,700000
```

### Expected Corpus Size

For meaningful ICF scores, you need:
- **Minimum**: 1M unique words, 1B total tokens
- **Recommended**: 10M+ unique words, 10B+ total tokens
- **Optimal**: 100M+ unique words, 100B+ total tokens (Google 1T scale)

The model will work with smaller datasets, but generalization (especially Jabberwocky Protocol) requires large-scale statistics.

