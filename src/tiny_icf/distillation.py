"""Knowledge Distillation from Language Models to Character-Level CNN.

This module implements knowledge distillation from pre-trained language models
(e.g., BERT, RoBERTa, sentence-transformers) to improve the character-level
CNN model's ICF prediction capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

# Try to import sentence-transformers (lightweight option)
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import transformers (for BERT/RoBERTa)
try:
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LanguageModelTeacher:
    """
    Wrapper for pre-trained language models used as teachers in distillation.

    Supports:
    - Sentence-transformers (lightweight, fast)
    - HuggingFace transformers (BERT, RoBERTa, etc.)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_type: str = "sentence-transformers",  # or "transformers"
        device: Optional[torch.device] = None,
        use_word_frequency_head: bool = False,
    ):
        """
        Args:
            model_name: Name of the pre-trained model
            model_type: "sentence-transformers" or "transformers"
            device: Device to run the model on
            use_word_frequency_head: If True, add a small head to predict ICF
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model_name = model_name
        self.use_word_frequency_head = use_word_frequency_head

        if model_type == "sentence-transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
            self.model = SentenceTransformer(model_name, device=str(self.device))
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        elif model_type == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers is required. Install with: pip install transformers"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            # Get embedding dimension from model config
            self.embedding_dim = self.model.config.hidden_size
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Optional: Add a small head to predict ICF from embeddings
        if use_word_frequency_head:
            self.icf_head = nn.Sequential(
                nn.Linear(self.embedding_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid(),  # ICF in [0, 1]
            ).to(self.device)
        else:
            self.icf_head = None

    def get_embeddings(self, words: list[str]) -> torch.Tensor:
        """
        Get embeddings for a list of words.

        Args:
            words: List of word strings

        Returns:
            [batch_size, embedding_dim] tensor of embeddings
        """
        with torch.no_grad():
            if self.model_type == "sentence-transformers":
                embeddings = self.model.encode(words, convert_to_tensor=True)
            else:  # transformers
                # Tokenize and encode
                inputs = self.tokenizer(
                    words,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)
                # Use [CLS] token embedding or mean pooling
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    # Mean pooling over sequence
                    embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings

    def predict_icf(self, words: list[str]) -> torch.Tensor:
        """
        Predict ICF scores using the teacher model (if head is enabled).

        Args:
            words: List of word strings

        Returns:
            [batch_size, 1] tensor of ICF predictions
        """
        if self.icf_head is None:
            raise ValueError("ICF head not enabled. Set use_word_frequency_head=True")

        embeddings = self.get_embeddings(words)
        with torch.no_grad():
            icf_scores = self.icf_head(embeddings)
        return icf_scores

    def get_intermediate_features(
        self, words: list[str], layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get intermediate layer features from the teacher model.

        Args:
            words: List of word strings
            layer_idx: Which layer to extract (None = last layer)

        Returns:
            [batch_size, feature_dim] tensor of features
        """
        if self.model_type != "transformers":
            # For sentence-transformers, we can only get final embeddings
            return self.get_embeddings(words)

        with torch.no_grad():
            inputs = self.tokenizer(
                words,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            # Get all hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            if layer_idx is None:
                # Use last layer
                features = hidden_states[-1]
            else:
                features = hidden_states[layer_idx]

            # Mean pooling over sequence length
            features = features.mean(dim=1)  # [batch_size, hidden_dim]

        return features


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    Combines:
    1. Standard supervised loss (hard targets)
    2. Distillation loss (soft targets from teacher)
    3. Feature alignment loss (optional)
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
        beta: float = 0.1,
        use_feature_distillation: bool = False,
        feature_projection_dim: Optional[int] = None,
        use_dynamic_temperature: bool = False,
        base_temperature: float = 3.0,
        min_temperature: float = 2.0,
        max_temperature: float = 10.0,
    ):
        """
        Args:
            temperature: Temperature for softening teacher predictions (static if use_dynamic_temperature=False)
            alpha: Weight for distillation loss (1-alpha for supervised loss)
            beta: Weight for feature alignment loss
            use_feature_distillation: Whether to align intermediate features
            feature_projection_dim: Dimension to project student features to match teacher
            use_dynamic_temperature: If True, adjust temperature based on student-teacher divergence
            base_temperature: Base temperature for dynamic scheduling
            min_temperature: Minimum temperature (clamp lower bound)
            max_temperature: Maximum temperature (clamp upper bound)
        """
        super().__init__()
        self.temperature = temperature
        self.use_dynamic_temperature = use_dynamic_temperature
        self.base_temperature = base_temperature if use_dynamic_temperature else temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.alpha = alpha  # Weight for distillation loss
        self.beta = beta  # Weight for feature alignment
        self.use_feature_distillation = use_feature_distillation

        # Projection layer to align student features with teacher features
        if use_feature_distillation and feature_projection_dim is not None:
            self.feature_projection = nn.Linear(feature_projection_dim, feature_projection_dim)
        else:
            self.feature_projection = None

    def compute_temperature(self, student_loss: torch.Tensor, teacher_loss: torch.Tensor) -> float:
        """
        Compute dynamic temperature based on student-teacher divergence.

        Higher divergence → higher temperature (softer guidance for stability)
        Lower divergence → lower temperature (sharper guidance for refinement)

        Args:
            student_loss: Current student loss value
            teacher_loss: Current teacher loss value (or reference loss)

        Returns:
            Adjusted temperature value
        """
        if not self.use_dynamic_temperature:
            return self.temperature

        # Compute divergence (normalized difference)
        divergence = torch.abs(student_loss - teacher_loss).item()
        if teacher_loss.item() > 0:
            normalized_divergence = divergence / teacher_loss.item()
        else:
            normalized_divergence = divergence

        # Adjust temperature: higher divergence → higher temperature
        # Formula: temp = base_temp * (1 + divergence_factor)
        # Clamp between min and max
        temp = self.base_temperature * (1.0 + normalized_divergence)
        temp = max(self.min_temperature, min(self.max_temperature, temp))

        return temp

    def forward(
        self,
        student_predictions: torch.Tensor,
        teacher_predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined distillation loss.

        Args:
            student_predictions: [batch_size, 1] student model predictions
            teacher_predictions: [batch_size, 1] teacher model predictions (soft targets)
            ground_truth: [batch_size, 1] ground truth ICF scores
            student_features: [batch_size, feature_dim] student intermediate features (optional)
            teacher_features: [batch_size, feature_dim] teacher intermediate features (optional)

        Returns:
            (total_loss, loss_components_dict)
        """
        # 1. Supervised loss (hard targets)
        supervised_loss = F.mse_loss(student_predictions, ground_truth)

        # Compute teacher loss for dynamic temperature (if enabled)
        teacher_loss = F.mse_loss(teacher_predictions, ground_truth)

        # 2. Distillation loss (soft targets with temperature)
        # Compute temperature (static or dynamic)
        current_temp = self.compute_temperature(supervised_loss, teacher_loss)

        # Soften teacher predictions
        teacher_soft = teacher_predictions / current_temp
        student_soft = student_predictions / current_temp

        # KL divergence between student and teacher distributions
        # For regression, we use MSE on softened predictions
        distillation_loss = F.mse_loss(student_soft, teacher_soft) * (current_temp**2)

        # 3. Feature alignment loss (optional)
        feature_loss = torch.tensor(0.0, device=student_predictions.device)
        if (
            self.use_feature_distillation
            and student_features is not None
            and teacher_features is not None
        ):
            if self.feature_projection is not None:
                student_features = self.feature_projection(student_features)

            # Align feature spaces using cosine similarity or MSE
            # Normalize features
            student_features_norm = F.normalize(student_features, p=2, dim=1)
            teacher_features_norm = F.normalize(teacher_features, p=2, dim=1)

            # Cosine similarity loss (maximize similarity)
            feature_loss = (
                1.0
                - F.cosine_similarity(student_features_norm, teacher_features_norm, dim=1).mean()
            )

        # Combine losses
        total_loss = (
            (1 - self.alpha) * supervised_loss
            + self.alpha * distillation_loss
            + self.beta * feature_loss
        )

        loss_components = {
            "supervised_loss": supervised_loss,
            "distillation_loss": distillation_loss,
            "feature_loss": feature_loss,
            "total_loss": total_loss,
            "temperature": torch.tensor(current_temp, device=student_predictions.device),
        }

        return total_loss, loss_components


class DistilledICFModel(nn.Module):
    """
    Wrapper that adds distillation capabilities to any ICF model.

    This allows distilling knowledge from a language model teacher
    into the character-level CNN student model.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: LanguageModelTeacher,
        feature_extraction_layer: Optional[str] = None,
    ):
        """
        Args:
            student_model: The character-level CNN model (UniversalICF, etc.)
            teacher_model: The language model teacher
            feature_extraction_layer: Which layer to extract features from student
                                     (e.g., 'conv_features', 'head_input')
        """
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.feature_extraction_layer = feature_extraction_layer

    def forward(
        self,
        byte_tensors: torch.Tensor,
        words: Optional[list[str]] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with distillation support.

        Args:
            byte_tensors: [batch_size, max_length] byte tensors for student
            words: List of word strings for teacher (optional, for distillation)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with:
            - 'student_predictions': Student model predictions
            - 'teacher_predictions': Teacher model predictions (if words provided)
            - 'student_features': Student intermediate features (if return_features)
            - 'teacher_features': Teacher features (if words provided and return_features)
        """
        # Student forward pass
        if hasattr(self.student_model, "forward") and hasattr(self.student_model, "forward"):
            # Check if model supports return_features
            try:
                student_output = self.student_model(byte_tensors, return_features=return_features)
                if return_features and isinstance(student_output, tuple):
                    student_predictions, features_dict = student_output
                    student_features = features_dict.get("feature_activations", None)
                else:
                    student_predictions = student_output
                    student_features = None
            except TypeError:
                # Model doesn't support return_features
                student_predictions = self.student_model(byte_tensors)
                student_features = None
        else:
            student_predictions = self.student_model(byte_tensors)
            student_features = None

        result = {
            "student_predictions": student_predictions,
        }

        if return_features and student_features is not None:
            result["student_features"] = student_features

        # Teacher forward pass (if words provided)
        if words is not None:
            with torch.no_grad():
                if self.teacher_model.use_word_frequency_head:
                    teacher_predictions = self.teacher_model.predict_icf(words)
                else:
                    # Use embeddings as proxy for ICF (higher embedding norm = more informative = higher ICF)
                    teacher_embeddings = self.teacher_model.get_embeddings(words)
                    # Normalize embeddings and use as soft ICF proxy
                    # This is a heuristic: more informative words have richer embeddings
                    teacher_predictions = torch.norm(teacher_embeddings, dim=1, keepdim=True)
                    teacher_predictions = (
                        teacher_predictions / teacher_predictions.max()
                    )  # Normalize to [0, 1]

                result["teacher_predictions"] = teacher_predictions

                if return_features:
                    teacher_features = self.teacher_model.get_intermediate_features(words)
                    result["teacher_features"] = teacher_features

        return result
