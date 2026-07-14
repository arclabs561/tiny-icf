"""Language detection using character n-grams and byte patterns."""

from typing import List, Dict, Tuple

# Language-specific character patterns
LANGUAGE_PATTERNS = {
    "en": {
        "chars": set("abcdefghijklmnopqrstuvwxyz"),
        "common_bigrams": {"th", "he", "in", "er", "an", "re", "ed", "nd", "on", "en"},
        "common_trigrams": {"the", "and", "ing", "ion", "ent", "for", "her", "ter", "tha", "ere"},
    },
    "es": {
        "chars": set("abcdefghijklmnopqrstuvwxyzáéíóúñü¿¡"),
        "common_bigrams": {"de", "la", "el", "en", "er", "ar", "es", "os", "as", "ra"},
        "common_trigrams": {"que", "del", "los", "las", "por", "una", "est", "con", "par", "ent"},
    },
    "fr": {
        "chars": set("abcdefghijklmnopqrstuvwxyzàâäéèêëîïôöùûüÿç"),
        "common_bigrams": {"le", "de", "er", "en", "on", "es", "re", "nt", "ou", "te"},
        "common_trigrams": {"les", "des", "ent", "ion", "que", "est", "eur", "ant", "ons", "ure"},
    },
    "de": {
        "chars": set("abcdefghijklmnopqrstuvwxyzäöüß"),
        "common_bigrams": {"er", "en", "ch", "de", "ie", "in", "un", "te", "nd", "st"},
        "common_trigrams": {"der", "die", "und", "den", "sch", "ich", "ein", "che", "ung", "gen"},
    },
    "it": {
        "chars": set("abcdefghijklmnopqrstuvwxyzàèéìíîòóùú"),
        "common_bigrams": {"er", "re", "en", "on", "in", "te", "ar", "es", "an", "or"},
        "common_trigrams": {"che", "del", "per", "una", "con", "ion", "ent", "est", "are", "ato"},
    },
    "pt": {
        "chars": set("abcdefghijklmnopqrstuvwxyzáâãàéêíóôõúç"),
        "common_bigrams": {"de", "er", "en", "ar", "os", "as", "es", "ra", "re", "te"},
        "common_trigrams": {"que", "dos", "das", "por", "uma", "est", "ent", "con", "par", "com"},
    },
    "ru": {
        "chars": set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
        "common_bigrams": {"ов", "ер", "ен", "ин", "ан", "ст", "ор", "ел", "ра", "то"},
        "common_trigrams": {"что", "это", "как", "для", "при", "про", "все", "был", "был", "его"},
    },
    "ko": {
        "chars": set("가나다라마바사아자차카타파하"),
        "common_bigrams": set(),  # Korean uses syllable blocks, not bigrams
        "common_trigrams": set(),
    },
    "zh": {
        "chars": set(
            "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"
        ),
        "common_bigrams": set(),
        "common_trigrams": set(),
    },
    "ja": {
        "chars": set(
            "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
        ),
        "common_bigrams": set(),
        "common_trigrams": set(),
    },
}


def extract_character_ngrams(word: str, n: int = 3) -> List[str]:
    """Extract character n-grams from word."""
    if len(word) < n:
        return [word]
    return [word[i : i + n] for i in range(len(word) - n + 1)]


def detect_language_simple(word: str) -> List[Tuple[str, float]]:
    """
    Simple language detection based on character patterns.

    Returns list of (language_code, confidence) tuples, sorted by confidence.
    """
    if not word or len(word.strip()) == 0:
        return [("en", 0.5)]  # Default

    word_lower = word.lower()

    # Check for language-specific characters
    scores = {}

    for lang, patterns in LANGUAGE_PATTERNS.items():
        score = 0.0

        # Character set match
        chars_in_word = set(word_lower)
        lang_chars = patterns.get("chars", set())
        if lang_chars:
            overlap = len(chars_in_word & lang_chars)
            total_chars = len(chars_in_word)
            if total_chars > 0:
                score += (overlap / total_chars) * 0.4

        # Bigram match
        bigrams = extract_character_ngrams(word_lower, 2)
        common_bigrams = patterns.get("common_bigrams", set())
        if common_bigrams:
            bigram_matches = sum(1 for bg in bigrams if bg in common_bigrams)
            if len(bigrams) > 0:
                score += (bigram_matches / len(bigrams)) * 0.3

        # Trigram match
        trigrams = extract_character_ngrams(word_lower, 3)
        common_trigrams = patterns.get("common_trigrams", set())
        if common_trigrams:
            trigram_matches = sum(1 for tg in trigrams if tg in common_trigrams)
            if len(trigrams) > 0:
                score += (trigram_matches / len(trigrams)) * 0.3

        if score > 0:
            scores[lang] = score

    # Normalize scores
    if not scores:
        return [("en", 0.5)]  # Default to English

    total_score = sum(scores.values())
    if total_score > 0:
        scores = {lang: score / total_score for lang, score in scores.items()}

    # Sort by confidence
    sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Return top languages with confidence > 0.1
    return [(lang, conf) for lang, conf in sorted_langs if conf > 0.1]


def detect_language_byte_patterns(word: str) -> List[Tuple[str, float]]:
    """
    Language detection based on byte-level patterns.

    Uses UTF-8 byte sequences to detect language encoding patterns.
    """
    if not word:
        return [("en", 0.5)]

    try:
        word_bytes = word.encode("utf-8")
    except (AttributeError, UnicodeEncodeError):
        return [("en", 0.5)]

    scores = {}

    # Check byte ranges for different scripts
    for byte_val in word_bytes:
        if 0x00 <= byte_val <= 0x7F:
            # ASCII - could be English, Spanish, French, German, etc.
            scores["en"] = scores.get("en", 0) + 0.2
            scores["es"] = scores.get("es", 0) + 0.15
            scores["fr"] = scores.get("fr", 0) + 0.15
            scores["de"] = scores.get("de", 0) + 0.1
        elif 0xC0 <= byte_val <= 0xFF:
            # Extended ASCII - likely Romance languages
            scores["es"] = scores.get("es", 0) + 0.3
            scores["fr"] = scores.get("fr", 0) + 0.3
            scores["pt"] = scores.get("pt", 0) + 0.2
            scores["it"] = scores.get("it", 0) + 0.2
        elif 0x80 <= byte_val <= 0xBF:
            # Continuation bytes - part of multi-byte sequence
            pass  # Already counted in lead byte
        elif 0xD0 <= byte_val <= 0xDF or 0xE0 <= byte_val <= 0xEF:
            # Cyrillic (Russian)
            scores["ru"] = scores.get("ru", 0) + 0.5
        elif 0xE0 <= byte_val <= 0xEF:
            # Could be Chinese, Japanese, Korean
            scores["zh"] = scores.get("zh", 0) + 0.3
            scores["ja"] = scores.get("ja", 0) + 0.2
            scores["ko"] = scores.get("ko", 0) + 0.2

    if not scores:
        return [("en", 0.5)]

    # Normalize
    total = sum(scores.values())
    if total > 0:
        scores = {lang: score / total for lang, score in scores.items()}

    sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(lang, conf) for lang, conf in sorted_langs if conf > 0.1]


def detect_languages(word: str, method: str = "combined") -> List[Tuple[str, float]]:
    """
    Detect likely languages for a word.

    Args:
        word: Word to analyze
        method: 'simple' (character patterns), 'byte' (byte patterns), or 'combined'

    Returns:
        List of (language_code, confidence) tuples, sorted by confidence (highest first)
    """
    if method == "simple":
        return detect_language_simple(word)
    elif method == "byte":
        return detect_language_byte_patterns(word)
    else:  # combined
        simple_results = detect_language_simple(word)
        byte_results = detect_language_byte_patterns(word)

        # Combine scores
        combined = {}
        for lang, conf in simple_results:
            combined[lang] = combined.get(lang, 0) + conf * 0.6
        for lang, conf in byte_results:
            combined[lang] = combined.get(lang, 0) + conf * 0.4

        if not combined:
            return [("en", 0.5)]

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {lang: score / total for lang, score in combined.items()}

        sorted_langs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [(lang, conf) for lang, conf in sorted_langs if conf > 0.1]


# Language code to name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ko": "Korean",
    "zh": "Chinese",
    "ja": "Japanese",
}


def format_languages(languages: List[Tuple[str, float]], top_k: int = 3) -> List[Dict[str, any]]:
    """
    Format language detection results.

    Returns:
        List of dicts with 'code', 'name', 'confidence'
    """
    results = []
    for lang_code, confidence in languages[:top_k]:
        results.append(
            {
                "code": lang_code,
                "name": LANGUAGE_NAMES.get(lang_code, lang_code.upper()),
                "confidence": float(confidence),
            }
        )
    return results
