# Universal Support: Symbols, Emojis, and Multilingual

## Extensions Implemented

### 1. ✅ Multilingual Typo Patterns

**File**: `src/tiny_icf/symbol_augmentation.py`

**Supported Languages**:
- **Spanish (es)**: á→a, é→e, í→i, ó→o, ú→u, ñ→n, ü→u
- **French (fr)**: à→a, â→a, é→e, è→e, ç→c, etc.
- **German (de)**: ä→a, ö→o, ü→u, ß→ss
- **Russian (ru)**: Cyrillic→Latin common mistakes

**Features**:
- Automatic language detection
- Language-specific typo patterns
- Accented→unaccented substitutions (common typos)

### 2. ✅ Random Symbols Support

**File**: `src/tiny_icf/symbol_augmentation.py`

**Handles**:
- Punctuation: `!`, `@`, `#`, `$`, `%`, `^`, `&`, `*`, etc.
- Special chars: `-`, `_`, `=`, `+`, `[`, `]`, `{`, `}`, etc.
- Web text patterns: symbols at end (usernames, tags)
- Symbol substitutions (typo-like)

**Augmentation**:
- Add symbols at word boundaries (common in web text)
- Replace characters with similar symbols
- Configurable probability (default: 5%)

### 3. ✅ Emoji/Emoticon Support

**File**: `src/tiny_icf/symbol_augmentation.py` + `scripts/download_emoji_frequencies.py`

**Handles**:
- **Unicode Emojis**: 😀, 😂, ❤️, 👍, etc. (50+ common ones)
- **Text Emoticons**: `:)`, `:(`, `;)`, `:D`, `:P`, `<3`, `xD`, etc.
- **Frequency Data**: Created default frequency list
- **Extraction**: Can extract from typo corpus

**Augmentation**:
- Add emojis at end (like "hello 😀")
- Emojis in middle (less common)
- Configurable probability (default: 2%)

### 4. ✅ Byte-Level Processing (Already Handles Everything)

**Key Insight**: Our byte-level CNN already handles:
- ✅ **All Unicode**: UTF-8 encoding covers all languages
- ✅ **Emojis**: Multi-byte sequences handled correctly
- ✅ **Symbols**: Any byte sequence works
- ✅ **Multilingual**: No language-specific tokenization needed

**Why This Works**:
- Input: Raw UTF-8 bytes (0-255)
- Model sees: Byte sequences, not characters
- Handles: Any language, emoji, symbol automatically

## Usage

### Universal Augmentation

```python
from tiny_icf.symbol_augmentation import UniversalAugmentation
from pathlib import Path

aug = UniversalAugmentation(
    typo_corpus_path=Path("data/typos/github_typos.csv"),
    symbol_prob=0.05,      # 5% symbol augmentation
    emoji_prob=0.02,        # 2% emoji augmentation
    multilingual_prob=0.1,  # 10% multilingual patterns
    keyboard_prob=0.15,     # 15% keyboard typos
)

# Works on any language/symbol/emoji
aug("hello")      # English
aug("café")       # French (might become "cafe")
aug("привет")     # Russian
aug("hello 😀")    # With emoji
```

### Universal Dataset

```python
from tiny_icf.data_universal import UniversalICFDataset, load_frequency_list_with_emojis

# Load with emojis
word_counts, total = load_frequency_list_with_emojis(
    Path("data/combined_frequencies.csv"),
    Path("data/emojis/emoji_frequencies.csv"),
)

# Create universal dataset
dataset = UniversalICFDataset(
    word_icf_pairs,
    typo_corpus_path=Path("data/typos/github_typos.csv"),
    emoji_freq_path=Path("data/emojis/emoji_frequencies.csv"),
    include_symbols=True,
    include_emojis=True,
)
```

## Data Created

### Emoji Frequencies
- `data/emojis/emoji_frequencies.csv`: 50+ emojis + text emoticons
- Based on common web usage patterns
- Can be extracted from typo corpus

## Why Byte-Level is Perfect

### Handles Everything Automatically

1. **Multilingual**: UTF-8 bytes encode all languages
   - Spanish: `café` → bytes `[99, 97, 102, 195, 169]`
   - Russian: `привет` → Cyrillic bytes
   - Chinese: `你好` → Multi-byte sequences

2. **Emojis**: Multi-byte UTF-8 sequences
   - `😀` → `[240, 159, 152, 128]` (4 bytes)
   - Model learns: "4-byte sequences = emojis"

3. **Symbols**: Any byte value (0-255)
   - `!@#$%` → byte sequences
   - Model learns: "non-alphanumeric bytes = symbols"

4. **No Tokenization Needed**: 
   - No language-specific rules
   - No emoji detection logic
   - Just raw bytes → model learns patterns

## Testing

```bash
# Test universal augmentation
python -c "
from tiny_icf.symbol_augmentation import UniversalAugmentation
from pathlib import Path

aug = UniversalAugmentation(
    Path('data/typos/github_typos.csv'),
    symbol_prob=1.0,
    emoji_prob=1.0,
)

words = ['hello', 'café', 'привет', 'test123']
for w in words:
    print(f'{w} -> {aug(w)}')
"
```

## Status

✅ **Multilingual**: Language-specific typo patterns
✅ **Symbols**: Random symbol augmentation
✅ **Emojis**: Emoji/emoticon support
✅ **Byte-Level**: Handles everything automatically
✅ **Frequency Data**: Emoji frequencies created

**The model is now truly universal!**

