"""Multi-task model architecture supporting ICF, language, temporal, and era predictions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

from tiny_icf.model import UniversalICF


class MultiTaskICF(nn.Module):
    """
    Multi-task model that predicts:
    - ICF score (regression)
    - Language (classification)
    - Era (classification)
    - Temporal ICF (optional, regression across decades)

    Shares base CNN architecture across all tasks.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        output_tasks: List[str] = ["icf"],
        vocab_size: int = 256,
        emb_dim: int = 36,
        conv_channels: int = 18,
        hidden_dim: int = 36,
        dropout: float = 0.4,
        num_languages: int = 10,
        num_eras: int = 5,
    ):
        """
        Args:
            base_model: Optional pre-trained base model (UniversalICF or ResidualICF)
            output_tasks: List of tasks to output ['icf', 'language', 'era', 'temporal']
            vocab_size: Byte vocabulary size (256 for UTF-8)
            emb_dim: Embedding dimension
            conv_channels: CNN channels
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            num_languages: Number of language classes
            num_eras: Number of era classes
        """
        super().__init__()

        # Use provided base model or create new one
        if base_model is not None:
            # Extract base CNN from existing model
            self.base = base_model
            # Get feature dimension from base model
            if hasattr(base_model, "head"):
                # UniversalICF: head input is conv_channels * 9
                feature_dim = conv_channels * 9
            else:
                # ResidualICF or other: need to infer
                feature_dim = hidden_dim
        else:
            # Create new base model
            self.base = UniversalICF(
                vocab_size=vocab_size,
                emb_dim=emb_dim,
                conv_channels=conv_channels,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            feature_dim = conv_channels * 9

        self.output_tasks = output_tasks

        # Task-specific heads
        if "icf" in output_tasks:
            # ICF head (regression)
            self.icf_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        if "language" in output_tasks:
            # Language head (classification)
            self.language_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_languages),
            )

        if "era" in output_tasks:
            # Era head (classification)
            self.era_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_eras),
            )

        if "temporal" in output_tasks:
            # Temporal head (regression for multiple decades)
            # Outputs ICF for each decade: [1800, 1900, 2000]
            self.temporal_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3),  # 3 decades
            )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features from base model."""
        # Get features from base model
        if isinstance(self.base, UniversalICF):
            # Use UniversalICF's internal structure to extract features
            # We need to extract features before the final head
            x_emb = self.base.emb(x)
            x_emb = x_emb.transpose(1, 2)

            c3 = F.relu(self.base.conv3(x_emb))
            c5 = F.relu(self.base.conv5(x_emb))
            c7 = F.relu(self.base.conv7(x_emb))

            # Multi-scale pooling (same as UniversalICF)
            p3_max = F.max_pool1d(c3, c3.size(2)).squeeze(2)
            p3_mean = F.avg_pool1d(c3, c3.size(2)).squeeze(2)
            p3_last = c3[:, :, -1]

            p5_max = F.max_pool1d(c5, c5.size(2)).squeeze(2)
            p5_mean = F.avg_pool1d(c5, c5.size(2)).squeeze(2)
            p5_last = c5[:, :, -1]

            p7_max = F.max_pool1d(c7, c7.size(2)).squeeze(2)
            p7_mean = F.avg_pool1d(c7, c7.size(2)).squeeze(2)
            p7_last = c7[:, :, -1]

            # Concatenate all features
            features = torch.cat(
                [p3_max, p3_mean, p3_last, p5_max, p5_mean, p5_last, p7_max, p7_mean, p7_last],
                dim=1,
            )
            return features
        else:
            # For other base models, try to use forward with return_features
            # This is a simplified version - may need model-specific handling
            if hasattr(self.base, "forward") and hasattr(self.base.forward, "__code__"):
                # Try calling with return_features if available
                try:
                    result = self.base(x, return_features=True)
                    if isinstance(result, tuple):
                        _, features_dict = result
                        if "feature_activations" in features_dict:
                            return features_dict["feature_activations"]
                except (TypeError, AttributeError, RuntimeError):
                    pass

            raise NotImplementedError(
                f"Base model {type(self.base)} not supported for feature extraction"
            )

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        Forward pass.

        Args:
            x: [Batch, Max_Char_Len] byte indices
            return_all: If True, return all task outputs; if False, return only ICF

        Returns:
            If return_all=False: [Batch, 1] ICF predictions
            If return_all=True: Dict with keys: 'icf', 'language', 'era', 'temporal'
        """
        # Extract shared features
        features = self._extract_features(x)

        outputs = {}

        # ICF prediction (always computed)
        if "icf" in self.output_tasks:
            icf_pred = self.icf_head(features)
            # Clip to [0, 1] range
            icf_pred = torch.clamp(icf_pred, 0.0, 1.0)
            outputs["icf"] = icf_pred

        # Language prediction
        if "language" in self.output_tasks:
            lang_logits = self.language_head(features)
            outputs["language"] = lang_logits

        # Era prediction
        if "era" in self.output_tasks:
            era_logits = self.era_head(features)
            outputs["era"] = era_logits

        # Temporal prediction
        if "temporal" in self.output_tasks:
            temporal_pred = self.temporal_head(features)
            # Clip to [0, 1] range
            temporal_pred = torch.clamp(temporal_pred, 0.0, 1.0)
            outputs["temporal"] = temporal_pred

        if return_all:
            return outputs
        else:
            # Return ICF by default (backward compatible)
            return outputs.get("icf", torch.zeros(x.size(0), 1, device=x.device))
