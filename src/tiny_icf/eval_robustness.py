# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24.0",
# ]
# ///
"""
Robustness testing for ICF predictions.

Implements:
- Adversarial examples (character perturbations)
- Out-of-distribution (OOD) testing
- Noise robustness (typo simulation)
- Character-level perturbations
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import random
import string


def character_perturbation(
    word: str,
    perturbation_type: str = "swap",
    n_perturbations: int = 1,
) -> List[str]:
    """
    Generate character-level perturbations of a word.

    Args:
        word: Input word
        perturbation_type: 'swap' (swap adjacent), 'delete' (delete char),
                          'insert' (insert char), 'substitute' (substitute char)
        n_perturbations: Number of perturbations to generate

    Returns:
        List of perturbed words
    """
    if not word:
        return []

    perturbed = []
    word_list = list(word)

    for _ in range(n_perturbations):
        if perturbation_type == "swap" and len(word_list) > 1:
            # Swap two adjacent characters
            idx = random.randint(0, len(word_list) - 2)
            word_copy = word_list.copy()
            word_copy[idx], word_copy[idx + 1] = word_copy[idx + 1], word_copy[idx]
            perturbed.append("".join(word_copy))

        elif perturbation_type == "delete" and len(word_list) > 1:
            # Delete a random character
            idx = random.randint(0, len(word_list) - 1)
            word_copy = word_list.copy()
            del word_copy[idx]
            perturbed.append("".join(word_copy))

        elif perturbation_type == "insert":
            # Insert a random character
            idx = random.randint(0, len(word_list))
            char = random.choice(string.ascii_lowercase)
            word_copy = word_list.copy()
            word_copy.insert(idx, char)
            perturbed.append("".join(word_copy))

        elif perturbation_type == "substitute" and len(word_list) > 0:
            # Substitute a random character
            idx = random.randint(0, len(word_list) - 1)
            char = random.choice(string.ascii_lowercase)
            word_copy = word_list.copy()
            word_copy[idx] = char
            perturbed.append("".join(word_copy))

    return perturbed


def typo_simulation(
    word: str,
    typo_rate: float = 0.1,
    typo_types: Optional[List[str]] = None,
) -> List[str]:
    """
    Simulate common typos in words.

    Args:
        word: Input word
        typo_rate: Probability of typo per character
        typo_types: List of typo types ('swap', 'delete', 'insert', 'substitute')

    Returns:
        List of words with typos
    """
    if typo_types is None:
        typo_types = ["swap", "substitute", "delete"]

    if not word:
        return []

    # Decide how many typos to introduce
    n_chars = len(word)
    expected_typos = int(n_chars * typo_rate)
    if expected_typos == 0:
        expected_typos = 1 if random.random() < typo_rate else 0

    if expected_typos == 0:
        return [word]

    # Generate typo
    typo_type = random.choice(typo_types)
    perturbed = character_perturbation(word, typo_type, n_perturbations=1)

    return perturbed if perturbed else [word]


def test_adversarial_robustness(
    model: Callable[[str], float],
    words: List[str],
    perturbation_types: Optional[List[str]] = None,
    n_perturbations_per_word: int = 5,
) -> Dict[str, float]:
    """
    Test model robustness to adversarial character perturbations.

    Args:
        model: Function that takes a word and returns ICF prediction
        words: List of test words
        perturbation_types: Types of perturbations to test
        n_perturbations_per_word: Number of perturbations per word

    Returns:
        Dictionary with robustness metrics
    """
    if perturbation_types is None:
        perturbation_types = ["swap", "delete", "insert", "substitute"]

    original_predictions = []
    perturbed_predictions = []
    prediction_diffs = []

    for word in words:
        if not word:
            continue

        # Original prediction
        orig_pred = model(word)
        original_predictions.append(orig_pred)

        # Generate perturbations
        for pert_type in perturbation_types:
            perturbations = character_perturbation(
                word, pert_type, n_perturbations=n_perturbations_per_word
            )

            for pert_word in perturbations:
                if pert_word != word:  # Skip if no change
                    pert_pred = model(pert_word)
                    perturbed_predictions.append(pert_pred)
                    prediction_diffs.append(abs(orig_pred - pert_pred))

    if not prediction_diffs:
        return {
            "mean_perturbation_error": 0.0,
            "max_perturbation_error": 0.0,
            "robustness_score": 1.0,
        }

    mean_error = np.mean(prediction_diffs)
    max_error = np.max(prediction_diffs)

    # Robustness score: lower error = more robust (normalized to [0, 1])
    # Assuming ICF is in [0, 1], max possible error is 1.0
    robustness_score = 1.0 - min(1.0, mean_error)

    return {
        "mean_perturbation_error": float(mean_error),
        "max_perturbation_error": float(max_error),
        "robustness_score": float(robustness_score),
        "n_perturbations": len(prediction_diffs),
    }


def test_ood_robustness(
    model: Callable[[str], float],
    ood_words: List[str],
    in_distribution_icf_range: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, float]:
    """
    Test model robustness on out-of-distribution (OOD) words.

    OOD words are those that are significantly different from training distribution.

    Args:
        model: Function that takes a word and returns ICF prediction
        ood_words: List of OOD words (e.g., gibberish, foreign words, code)
        in_distribution_icf_range: Expected ICF range for in-distribution words

    Returns:
        Dictionary with OOD robustness metrics
    """
    ood_predictions = []

    for word in ood_words:
        if not word:
            continue
        pred = model(word)
        ood_predictions.append(pred)

    if not ood_predictions:
        return {
            "mean_ood_icf": 0.0,
            "ood_detection_rate": 0.0,
        }

    ood_predictions = np.array(ood_predictions)

    # OOD words should have high ICF (rare/gibberish)
    # Detection rate: how many OOD words are predicted as rare (ICF > threshold)
    threshold = 0.7  # Consider ICF > 0.7 as rare/OOD
    ood_detection_rate = np.mean(ood_predictions > threshold)

    return {
        "mean_ood_icf": float(ood_predictions.mean()),
        "std_ood_icf": float(ood_predictions.std()),
        "ood_detection_rate": float(ood_detection_rate),
        "n_ood_words": len(ood_predictions),
    }


def test_noise_robustness(
    model: Callable[[str], float],
    words: List[str],
    noise_levels: Optional[List[float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Test model robustness to noise (typos).

    Args:
        model: Function that takes a word and returns ICF prediction
        words: List of test words
        noise_levels: List of typo rates to test (e.g., [0.1, 0.2, 0.3])

    Returns:
        Dictionary mapping noise level to robustness metrics
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.2, 0.3]

    results = {}

    for noise_level in noise_levels:
        original_predictions = []
        noisy_predictions = []
        prediction_diffs = []

        for word in words:
            if not word:
                continue

            # Original prediction
            orig_pred = model(word)
            original_predictions.append(orig_pred)

            # Generate noisy version
            noisy_words = typo_simulation(word, typo_rate=noise_level)
            for noisy_word in noisy_words:
                if noisy_word != word:
                    noisy_pred = model(noisy_word)
                    noisy_predictions.append(noisy_pred)
                    prediction_diffs.append(abs(orig_pred - noisy_pred))

        if prediction_diffs:
            results[f"noise_{noise_level}"] = {
                "mean_error": float(np.mean(prediction_diffs)),
                "max_error": float(np.max(prediction_diffs)),
                "robustness_score": float(1.0 - min(1.0, np.mean(prediction_diffs))),
                "n_noisy_samples": len(prediction_diffs),
            }
        else:
            results[f"noise_{noise_level}"] = {
                "mean_error": 0.0,
                "max_error": 0.0,
                "robustness_score": 1.0,
                "n_noisy_samples": 0,
            }

    return results


def compute_robustness_metrics(
    model: Callable[[str], float],
    test_words: List[str],
    ood_words: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Compute comprehensive robustness metrics.

    Args:
        model: Function that takes a word and returns ICF prediction
        test_words: List of in-distribution test words
        ood_words: Optional list of OOD words

    Returns:
        Dictionary with all robustness metrics
    """
    results = {}

    # Adversarial robustness
    adversarial = test_adversarial_robustness(model, test_words)
    results["adversarial"] = adversarial

    # Noise robustness
    noise = test_noise_robustness(model, test_words)
    results["noise"] = noise

    # OOD robustness
    if ood_words:
        ood = test_ood_robustness(model, ood_words)
        results["ood"] = ood

    # Overall robustness score (weighted average)
    adversarial_score = adversarial.get("robustness_score", 0.0)
    noise_scores = [v.get("robustness_score", 0.0) for v in noise.values()]
    noise_score = np.mean(noise_scores) if noise_scores else 0.0

    overall_score = (adversarial_score + noise_score) / 2.0
    if ood_words and "ood" in results:
        ood_score = results["ood"].get("ood_detection_rate", 0.0)
        overall_score = (adversarial_score + noise_score + ood_score) / 3.0

    results["overall_robustness"] = float(overall_score)

    return results
