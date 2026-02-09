"""Isotonic regret text reduction: Remove words one at a time, track regret progression.

The isotonic property: Regret should increase monotonically as we remove words.
This ensures we're making progress and not oscillating.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def compute_embedding_difference(
    original_embedding: torch.Tensor,
    reduced_embedding: torch.Tensor,
) -> float:
    """Compute embedding difference (regret) using cosine distance."""
    cos_sim = torch.nn.functional.cosine_similarity(
        original_embedding.unsqueeze(0),
        reduced_embedding.unsqueeze(0),
    ).item()
    regret = 1.0 - cos_sim
    return regret


class IsotonicTextReducer:
    """
    Text reducer with isotonic regret tracking.

    Removes words one at a time, tracking regret at each step.
    Ensures regret increases monotonically (isotonic property).
    """

    def __init__(
        self,
        icf_model,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            icf_model: Trained ICF prediction model
            embedding_model_name: Sentence transformer model name
            device: Device for computation
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        self.icf_model = icf_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=str(self.device))

    def predict_icf(self, word: str) -> float:
        """Predict ICF for a word."""
        byte_seq = word.encode("utf-8")[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            icf = self.icf_model(byte_tensor).item()

        return icf

    def compute_embedding(self, text: str) -> torch.Tensor:
        """Compute embedding for text."""
        embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        return embedding

    def reduce_isotonic(
        self,
        text: str,
        target_ratio: float = 0.5,
        enforce_isotonic: bool = True,
        verbose: bool = False,
    ) -> Tuple[str, float, Dict]:
        """
        Reduce text with isotonic regret tracking.

        Removes words one at a time, tracking regret at each step.
        Ensures regret increases monotonically (isotonic property).

        Args:
            text: Input text
            target_ratio: Fraction of words to keep
            enforce_isotonic: If True, ensure regret increases monotonically
            verbose: Print progress at each step

        Returns:
            (reduced_text, final_regret, stats)
            stats includes:
            - progression: List of (step, words_remaining, regret, word_removed, icf_removed)
            - regret_curve: Regret values at each step
            - is_isotonic: Whether regret increased monotonically
        """
        words = text.split()
        n_words = len(words)
        target_length = max(1, int(n_words * target_ratio))

        if n_words <= target_length:
            return (
                text,
                0.0,
                {
                    "original_length": n_words,
                    "reduced_length": n_words,
                    "reduction_ratio": 0.0,
                    "regret": 0.0,
                    "progression": [],
                    "regret_curve": [0.0],
                    "is_isotonic": True,
                },
            )

        # Compute original embedding once
        original_embedding = self.compute_embedding(text)

        # Predict ICF for all words
        icf_scores = [self.predict_icf(word) for word in words]

        # Track progression
        progression = []
        regret_curve = [0.0]  # Start with 0 regret (no words removed)

        current_words = words.copy()
        current_icf = icf_scores.copy()
        previous_regret = 0.0

        step = 0
        while len(current_words) > target_length:
            min_regret = float("inf")
            best_idx = -1
            best_regret = None

            # Try dropping each word, find one with least regret
            for i in range(len(current_words)):
                # Try dropping word i
                test_words = current_words[:i] + current_words[i + 1 :]
                test_text = " ".join(test_words)
                test_embedding = self.compute_embedding(test_text)
                regret = compute_embedding_difference(original_embedding, test_embedding)

                # Weight by ICF: prefer dropping low ICF words
                weighted_regret = regret * (1.0 - current_icf[i])

                # Enforce isotonic property: regret should only increase
                if enforce_isotonic and regret < previous_regret:
                    # Skip if this would decrease regret (violates isotonic property)
                    continue

                if weighted_regret < min_regret:
                    min_regret = weighted_regret
                    best_idx = i
                    best_regret = regret

            # If no valid candidate (all violate isotonic), pick lowest ICF
            if best_idx < 0:
                # Fallback: drop lowest ICF word
                min_icf_idx = min(range(len(current_icf)), key=lambda i: current_icf[i])
                best_idx = min_icf_idx
                test_words = current_words[:best_idx] + current_words[best_idx + 1 :]
                test_text = " ".join(test_words)
                test_embedding = self.compute_embedding(test_text)
                best_regret = compute_embedding_difference(original_embedding, test_embedding)

            # Check isotonic property
            is_isotonic_step = best_regret >= previous_regret

            # Record progression
            word_removed = current_words[best_idx]
            icf_removed = current_icf[best_idx]
            progression.append(
                {
                    "step": step,
                    "words_remaining": len(current_words) - 1,
                    "regret": best_regret,
                    "regret_delta": best_regret - previous_regret,
                    "word_removed": word_removed,
                    "icf_removed": icf_removed,
                    "is_isotonic": is_isotonic_step,
                }
            )
            regret_curve.append(best_regret)

            if verbose:
                print(
                    f"Step {step}: Removed '{word_removed}' (ICF={icf_removed:.3f}), "
                    f"regret={best_regret:.4f} (Δ={best_regret - previous_regret:+.4f}), "
                    f"words={len(current_words) - 1}/{n_words}"
                )

            # Drop the word
            current_words.pop(best_idx)
            current_icf.pop(best_idx)
            previous_regret = best_regret

            step += 1

        reduced_text = " ".join(current_words)

        # Compute final regret
        final_embedding = self.compute_embedding(reduced_text)
        final_regret = compute_embedding_difference(original_embedding, final_embedding)

        # Check if entire progression is isotonic
        is_isotonic = all(
            regret_curve[i] >= regret_curve[i - 1] for i in range(1, len(regret_curve))
        )

        stats = {
            "original_length": n_words,
            "reduced_length": len(current_words),
            "reduction_ratio": 1.0 - (len(current_words) / n_words),
            "avg_icf_kept": np.mean(current_icf) if current_icf else 0.0,
            "avg_icf_removed": (
                np.mean([p["icf_removed"] for p in progression]) if progression else 0.0
            ),
            "regret": final_regret,
            "steps": step,
            "progression": progression,
            "regret_curve": regret_curve,
            "is_isotonic": is_isotonic,
            "max_regret_increase": max(
                (regret_curve[i] - regret_curve[i - 1] for i in range(1, len(regret_curve))),
                default=0.0,
            ),
            "min_regret_increase": min(
                (regret_curve[i] - regret_curve[i - 1] for i in range(1, len(regret_curve))),
                default=0.0,
            ),
        }

        return reduced_text, final_regret, stats


def reduce_text_isotonic(
    text: str,
    icf_model,
    target_ratio: float = 0.5,
    embedding_model: str = "all-MiniLM-L6-v2",
    device: Optional[torch.device] = None,
    enforce_isotonic: bool = True,
    verbose: bool = False,
) -> Tuple[str, float, Dict]:
    """
    Reduce text with isotonic regret tracking.

    Args:
        text: Input text
        icf_model: ICF prediction model
        target_ratio: Fraction of words to keep
        embedding_model: Sentence transformer model name
        device: Device for computation
        enforce_isotonic: Ensure regret increases monotonically
        verbose: Print progress

    Returns:
        (reduced_text, regret, stats)
    """
    reducer = IsotonicTextReducer(icf_model, embedding_model, device)
    return reducer.reduce_isotonic(text, target_ratio, enforce_isotonic, verbose)
