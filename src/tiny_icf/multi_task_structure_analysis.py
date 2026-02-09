"""Analyze structure across ALL tasks, not just ICF prediction.

Tasks:
1. ICF Prediction: word → ICF score
2. Text Reduction: word → embedding regret (drop words optimally)
3. Temporal ICF: word → ICF across decades (1800s, 1900s, 2000s)
4. Language Detection: word → language probabilities
5. Era Classification: word → historical era (archaic, modern, contemporary)
6. Multi-objective: All tasks combined with AMOO

For each task, analyze:
- Structure strength (can we compress?)
- Generalization potential (can model learn patterns?)
- Kolmogorov complexity constraints
- Compression vs dictionary trade-offs
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

try:
    from scipy import stats
    from scipy.stats import entropy

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def analyze_icf_structure(word_icf: Dict[str, float]) -> Dict[str, float]:
    """Analyze structure in ICF prediction task."""
    results = {
        "task": "ICF Prediction",
        "vocabulary_size": len(word_icf),
    }

    # N-gram correlation
    ngram_correlations = {}
    for n in [2, 3, 4]:
        ngram_to_icfs = defaultdict(list)
        for word, icf in word_icf.items():
            for i in range(len(word) - n + 1):
                ngram = word[i : i + n].lower()
                ngram_to_icfs[ngram].append(icf)

        ngram_avg = {ngram: np.mean(icfs) for ngram, icfs in ngram_to_icfs.items() if len(icfs) > 1}
        ngram_freq = {ngram: len(ngram_to_icfs[ngram]) for ngram in ngram_avg.keys()}

        if len(ngram_freq) > 10 and HAS_SCIPY:
            freqs = list(ngram_freq.values())
            avg_icfs = [ngram_avg[ngram] for ngram in ngram_freq.keys()]
            corr, p_val = stats.pearsonr(freqs, avg_icfs)
            ngram_correlations[f"ngram_{n}"] = corr
            results[f"ngram_{n}_correlation"] = corr
            results[f"ngram_{n}_p_value"] = p_val

    # Shannon entropy
    icf_values = np.array(list(word_icf.values()))
    icf_probs = np.histogram(icf_values, bins=20, density=True)[0]
    icf_probs = icf_probs / icf_probs.sum()
    if HAS_SCIPY:
        h_icf = entropy(icf_probs[icf_probs > 0], base=2)
        results["shannon_entropy"] = h_icf
        results["compression_potential"] = (
            (len(word_icf) * 32) / (len(word_icf) * h_icf) if h_icf > 0 else 0
        )

    # Structure strength
    avg_corr = np.mean(list(ngram_correlations.values())) if ngram_correlations else 0
    results["structure_strength"] = abs(avg_corr)
    results["interpretation"] = (
        "strong" if abs(avg_corr) > 0.3 else "moderate" if abs(avg_corr) > 0.1 else "weak"
    )

    return results


def analyze_text_reduction_structure(
    word_icf: Dict[str, float],
    embedding_regret_data: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Analyze structure in text reduction task.

    Text reduction: Drop words to minimize embedding regret.
    Structure: Words with similar ICF should have similar regret.
    """
    results = {
        "task": "Text Reduction",
        "vocabulary_size": len(word_icf),
    }

    if embedding_regret_data is None:
        # Simulate: regret ≈ ICF (rare words have higher regret when dropped)
        embedding_regret_data = {
            word: icf * 0.8 + np.random.normal(0, 0.1) for word, icf in word_icf.items()
        }
        results["note"] = "Using simulated regret data"

    # Correlation: ICF vs regret
    words_common = list(set(word_icf.keys()) & set(embedding_regret_data.keys()))
    if len(words_common) > 10 and HAS_SCIPY:
        icfs = [word_icf[w] for w in words_common]
        regrets = [embedding_regret_data[w] for w in words_common]
        corr, p_val = stats.pearsonr(icfs, regrets)
        results["icf_regret_correlation"] = corr
        results["icf_regret_p_value"] = p_val
        results["structure_strength"] = abs(corr)
        results["interpretation"] = (
            "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
        )
    else:
        results["structure_strength"] = 0.0
        results["interpretation"] = "unknown"

    return results


def analyze_temporal_icf_structure(
    word_icf: Dict[str, float],
    temporal_data: Optional[Dict[str, Dict[int, float]]] = None,
) -> Dict[str, float]:
    """
    Analyze structure in temporal ICF prediction.

    Temporal ICF: Predict ICF across decades (1800s, 1900s, 2000s).
    Structure: ICF should change smoothly over time, similar words have similar temporal patterns.
    """
    results = {
        "task": "Temporal ICF Prediction",
        "vocabulary_size": len(word_icf),
    }

    if temporal_data is None:
        # Simulate: ICF changes over time (some words become more/less common)
        temporal_data = {}
        for word, current_icf in list(word_icf.items())[:1000]:  # Sample
            # Simulate temporal change
            temporal_data[word] = {
                1800: current_icf + np.random.normal(0, 0.2),
                1900: current_icf + np.random.normal(0, 0.15),
                2000: current_icf + np.random.normal(0, 0.1),
            }
        results["note"] = "Using simulated temporal data"

    # Analyze temporal consistency
    # Words with similar current ICF should have similar temporal patterns
    if len(temporal_data) > 10:
        # Group words by current ICF (bins)
        icf_bins = defaultdict(list)
        for word, decades in temporal_data.items():
            if word in word_icf:
                bin_idx = int(word_icf[word] * 10)  # 10 bins
                icf_bins[bin_idx].append(word)

        # Compute temporal variance within bins
        temporal_variances = []
        for bin_words in icf_bins.values():
            if len(bin_words) < 2:
                continue
            variances = []
            for word in bin_words:
                if word in temporal_data:
                    decade_icfs = list(temporal_data[word].values())
                    variances.append(np.var(decade_icfs))
            if variances:
                temporal_variances.append(np.mean(variances))

        if temporal_variances:
            results["temporal_consistency"] = 1.0 / (1.0 + np.mean(temporal_variances))
            results["structure_strength"] = results["temporal_consistency"]
            results["interpretation"] = (
                "strong"
                if results["temporal_consistency"] > 0.7
                else "moderate" if results["temporal_consistency"] > 0.4 else "weak"
            )
        else:
            results["structure_strength"] = 0.0
            results["interpretation"] = "unknown"
    else:
        results["structure_strength"] = 0.0
        results["interpretation"] = "insufficient_data"

    return results


def analyze_language_detection_structure(
    word_icf: Dict[str, float],
    language_data: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Analyze structure in language detection task.

    Language detection: Predict language from character patterns.
    Structure: Character n-grams strongly indicate language.
    """
    results = {
        "task": "Language Detection",
        "vocabulary_size": len(word_icf),
    }

    if language_data is None:
        # Simulate: Use character patterns to infer language
        # Common patterns: 'ing', 'tion' → English, 'ción', 'mente' → Spanish, etc.
        language_data = {}
        for word in list(word_icf.keys())[:1000]:  # Sample
            if "ing" in word or "tion" in word:
                language_data[word] = "en"
            elif "ción" in word or "mente" in word:
                language_data[word] = "es"
            elif "tion" in word and "e" in word[-3:]:
                language_data[word] = "fr"
            else:
                language_data[word] = "en"  # Default
        results["note"] = "Using simulated language data"

    # Analyze: Do character patterns predict language?
    # N-gram → language mapping
    ngram_to_langs = defaultdict(lambda: defaultdict(int))
    for word, lang in language_data.items():
        for n in [3, 4]:
            for i in range(len(word) - n + 1):
                ngram = word[i : i + n].lower()
                ngram_to_langs[ngram][lang] += 1

    # Compute n-gram language specificity
    ngram_specificities = []
    for ngram, lang_counts in ngram_to_langs.items():
        if len(lang_counts) > 1:
            # Entropy: higher = less specific, lower = more specific
            total = sum(lang_counts.values())
            probs = [count / total for count in lang_counts.values()]
            if HAS_SCIPY:
                h = entropy(probs, base=2)
                specificity = 1.0 - (h / math.log(len(lang_counts), 2))  # Normalize
                ngram_specificities.append(specificity)

    if ngram_specificities:
        results["avg_ngram_specificity"] = np.mean(ngram_specificities)
        results["structure_strength"] = results["avg_ngram_specificity"]
        results["interpretation"] = (
            "strong"
            if results["avg_ngram_specificity"] > 0.7
            else "moderate" if results["avg_ngram_specificity"] > 0.4 else "weak"
        )
    else:
        results["structure_strength"] = 0.0
        results["interpretation"] = "unknown"

    return results


def analyze_era_classification_structure(
    word_icf: Dict[str, float],
    era_data: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Analyze structure in era classification task.

    Era classification: Predict historical era (archaic, modern, contemporary).
    Structure: Character patterns and word structure indicate era.
    """
    results = {
        "task": "Era Classification",
        "vocabulary_size": len(word_icf),
    }

    if era_data is None:
        # Simulate: Use word patterns to infer era
        # Archaic: 'thou', 'thee', 'hast' → short, old patterns
        # Modern: longer words, technical terms
        era_data = {}
        for word in list(word_icf.keys())[:1000]:  # Sample
            if len(word) <= 4 and word.startswith("th"):
                era_data[word] = "archaic"
            elif "tech" in word or "cyber" in word or "selfie" in word:
                era_data[word] = "contemporary"
            elif len(word) > 8:
                era_data[word] = "modern"
            else:
                era_data[word] = "modern"  # Default
        results["note"] = "Using simulated era data"

    # Analyze: Do patterns predict era?
    # Similar to language detection
    pattern_to_eras = defaultdict(lambda: defaultdict(int))
    for word, era in era_data.items():
        # Use word length and character patterns
        pattern = f"len_{len(word)}"
        pattern_to_eras[pattern][era] += 1

        # Character patterns
        for n in [2, 3]:
            for i in range(len(word) - n + 1):
                ngram = word[i : i + n].lower()
                pattern_to_eras[ngram][era] += 1

    pattern_specificities = []
    for pattern, era_counts in pattern_to_eras.items():
        if len(era_counts) > 1:
            total = sum(era_counts.values())
            probs = [count / total for count in era_counts.values()]
            if HAS_SCIPY:
                h = entropy(probs, base=2)
                specificity = 1.0 - (h / math.log(len(era_counts), 2))
                pattern_specificities.append(specificity)

    if pattern_specificities:
        results["avg_pattern_specificity"] = np.mean(pattern_specificities)
        results["structure_strength"] = results["avg_pattern_specificity"]
        results["interpretation"] = (
            "strong"
            if results["avg_pattern_specificity"] > 0.7
            else "moderate" if results["avg_pattern_specificity"] > 0.4 else "weak"
        )
    else:
        results["structure_strength"] = 0.0
        results["interpretation"] = "unknown"

    return results


def analyze_multi_task_structure(
    task_results: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Analyze structure across all tasks (multi-objective).

    Key question: Do tasks share structure? Can one model learn all tasks?
    """
    results = {
        "task": "Multi-Task (All Tasks Combined)",
        "num_tasks": len(task_results),
    }

    # Average structure strength across tasks
    strengths = [r.get("structure_strength", 0) for r in task_results]
    results["avg_structure_strength"] = np.mean(strengths)
    results["min_structure_strength"] = np.min(strengths)
    results["max_structure_strength"] = np.max(strengths)

    # Task compatibility: Do tasks have similar structure?
    # If all tasks have strong structure, multi-task learning is feasible
    strong_tasks = sum(1 for s in strengths if s > 0.7)
    moderate_tasks = sum(1 for s in strengths if 0.4 < s <= 0.7)
    weak_tasks = sum(1 for s in strengths if s <= 0.4)

    results["strong_tasks"] = strong_tasks
    results["moderate_tasks"] = moderate_tasks
    results["weak_tasks"] = weak_tasks

    # Multi-task feasibility
    if strong_tasks + moderate_tasks >= len(task_results) * 0.7:
        results["multi_task_feasible"] = True
        results["interpretation"] = "feasible"
    elif strong_tasks + moderate_tasks >= len(task_results) * 0.5:
        results["multi_task_feasible"] = True
        results["interpretation"] = "moderate"
    else:
        results["multi_task_feasible"] = False
        results["interpretation"] = "challenging"

    return results


def analyze_all_tasks(
    word_icf: Dict[str, float],
    embedding_regret: Optional[Dict[str, float]] = None,
    temporal_data: Optional[Dict[str, Dict[int, float]]] = None,
    language_data: Optional[Dict[str, str]] = None,
    era_data: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze structure across ALL tasks.

    Returns:
        Dictionary mapping task names to their structure analysis results
    """
    results = {}

    # 1. ICF Prediction
    print("Analyzing ICF Prediction structure...")
    results["icf_prediction"] = analyze_icf_structure(word_icf)

    # 2. Text Reduction
    print("Analyzing Text Reduction structure...")
    results["text_reduction"] = analyze_text_reduction_structure(word_icf, embedding_regret)

    # 3. Temporal ICF
    print("Analyzing Temporal ICF structure...")
    results["temporal_icf"] = analyze_temporal_icf_structure(word_icf, temporal_data)

    # 4. Language Detection
    print("Analyzing Language Detection structure...")
    results["language_detection"] = analyze_language_detection_structure(word_icf, language_data)

    # 5. Era Classification
    print("Analyzing Era Classification structure...")
    results["era_classification"] = analyze_era_classification_structure(word_icf, era_data)

    # 6. Multi-Task
    print("Analyzing Multi-Task structure...")
    task_results = [
        results[k]
        for k in [
            "icf_prediction",
            "text_reduction",
            "temporal_icf",
            "language_detection",
            "era_classification",
        ]
    ]
    results["multi_task"] = analyze_multi_task_structure(task_results)

    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from tiny_icf.data import load_frequency_list, compute_normalized_icf
    except ImportError:
        print("Warning: Could not import tiny_icf.data, using simple loader")

        def load_frequency_list(filepath):
            import csv

            word_counts = {}
            total_tokens = 0
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if first_row and (
                    first_row[0].lower() in ["word", "token"] or not first_row[1].isdigit()
                ):
                    pass
                else:
                    if first_row and len(first_row) >= 2:
                        try:
                            word = first_row[0].strip().lower()
                            count = int(first_row[1])
                            word_counts[word] = word_counts.get(word, 0) + count
                            total_tokens += count
                        except (ValueError, IndexError):
                            pass
                for row in reader:
                    if len(row) < 2:
                        continue
                    try:
                        word = row[0].strip().lower()
                        count = int(row[1])
                        word_counts[word] = word_counts.get(word, 0) + count
                        total_tokens += count
                    except (ValueError, IndexError):
                        continue
            return word_counts, total_tokens

        def compute_normalized_icf(word_counts, total_tokens):
            import math

            log_total = math.log(total_tokens + 1)
            icf_scores = {}
            for word, count in word_counts.items():
                if count < 5:
                    icf_scores[word] = 1.0
                elif count >= total_tokens:
                    icf_scores[word] = 0.0
                else:
                    icf = math.log((total_tokens + 1) / (count + 1)) / log_total
                    icf_scores[word] = max(0.0, min(1.0, icf))
            return icf_scores

    # Find data file
    data_paths = [
        Path("data/word_frequency.csv"),
        Path("data/combined_frequencies.csv"),
        Path("data/word_frequency_modern.csv"),
    ]

    data_path = None
    for path in data_paths:
        if path.exists():
            data_path = path
            break

    if data_path:
        print(f"Loading data from: {data_path}")
        word_counts, total_tokens = load_frequency_list(data_path)
        word_icf = compute_normalized_icf(word_counts, total_tokens)

        print(f"\nLoaded {len(word_icf):,} words, {total_tokens:,} total tokens")
        print("=" * 70)

        # Analyze all tasks
        all_results = analyze_all_tasks(word_icf)

        # Print results
        print("\n" + "=" * 70)
        print("STRUCTURE ANALYSIS: ALL TASKS")
        print("=" * 70)

        for task_name, task_results in all_results.items():
            print(f"\n### {task_results.get('task', task_name).upper()}")
            print(
                f"  Structure Strength: {task_results.get('structure_strength', 0):.3f} ({task_results.get('interpretation', 'unknown')})"
            )
            if "vocabulary_size" in task_results:
                print(f"  Vocabulary Size: {task_results['vocabulary_size']:,}")
            if "multi_task_feasible" in task_results:
                print(f"  Multi-Task Feasible: {task_results['multi_task_feasible']}")
                print(f"  Strong Tasks: {task_results.get('strong_tasks', 0)}")
                print(f"  Moderate Tasks: {task_results.get('moderate_tasks', 0)}")
                print(f"  Weak Tasks: {task_results.get('weak_tasks', 0)}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        multi_task = all_results["multi_task"]
        print(f"\nMulti-Task Learning: {multi_task.get('interpretation', 'unknown')}")
        print(f"Average Structure Strength: {multi_task.get('avg_structure_strength', 0):.3f}")
        print(
            f"  Range: [{multi_task.get('min_structure_strength', 0):.3f}, {multi_task.get('max_structure_strength', 0):.3f}]"
        )

        if multi_task.get("multi_task_feasible", False):
            print("\n✓ Multi-task learning is FEASIBLE")
            print("  Unified model can learn all tasks")
        else:
            print("\n⚠️  Multi-task learning is CHALLENGING")
            print("  Consider task-specific models or stronger regularization")
    else:
        print("No data file found. Tried:")
        for path in data_paths:
            print(f"  - {path}")
