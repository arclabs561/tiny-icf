"""Multi-task model architecture supporting ICF + auxiliary heads.

This module is intentionally *optional*: the default tiny-icf CLI trains an ICF-only
`UniversalICF`. When you want richer downstream signals, you can wrap a base model
and add heads for:
- language (classification)
- era (classification)
- temporal ICF (regression across decades)
- token hygiene (classification: word vs url/email/code/etc)

Design constraint:
- Keep ICF behavior identical to the base model (including `return_features=True`)
  so OOV calibration and downstream scripts remain compatible.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Sequence, Any

from tiny_icf.model import UniversalICF


class MultiTaskICF(nn.Module):
    """
    Multi-task model that predicts:
    - ICF score (regression)
    - Language (classification)
    - Era (classification)
    - Temporal ICF (optional, regression across decades)
    - Token hygiene (optional, classification)

    Shares base CNN architecture across all tasks.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        output_tasks: List[str] = ["icf"],
        base_model_kwargs: Optional[Dict[str, Any]] = None,
        num_languages: int = 10,
        num_eras: int = 5,
        num_hygiene: int = 8,
        temporal_decades: Optional[Sequence[int]] = (1800, 1900, 2000),
    ):
        """
        Args:
            base_model: Optional pre-trained base model (UniversalICF or ResidualICF)
            output_tasks: List of tasks to output ['icf', 'language', 'era', 'temporal']
            base_model_kwargs: Optional kwargs for constructing the base model when base_model=None.
            num_languages: Number of language classes
            num_eras: Number of era classes
            num_hygiene: Number of hygiene classes
            temporal_decades: Decades to predict ICF for when 'temporal' is enabled.
        """
        super().__init__()

        # Use provided base model or create new one.
        if base_model is None:
            base_model_kwargs = dict(base_model_kwargs or {})
            self.base: nn.Module = UniversalICF(**base_model_kwargs)
        else:
            self.base = base_model

        self.output_tasks = output_tasks
        self.num_languages = int(num_languages)
        self.num_eras = int(num_eras)
        self.num_hygiene = int(num_hygiene)
        self.temporal_decades = tuple(int(d) for d in (temporal_decades or ()))

        # Feature dimension: prefer base return_features feature_activations (hidden_dim).
        # UniversalICF exposes this; other bases may not.
        if isinstance(self.base, UniversalICF):
            # `head[0]` is the first Linear(conv_channels*9 -> hidden_dim)
            feature_dim = int(getattr(self.base.head[0], "out_features", 36))
            dropout = float(getattr(self.base.head[2], "p", 0.4))
        else:
            # Conservative fallback (works if base return_features yields feature_activations).
            feature_dim = 36
            dropout = 0.4

        # Task-specific heads
        if "language" in output_tasks:
            # Language head (classification)
            self.language_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_languages),
            )

        if "era" in output_tasks:
            # Era head (classification)
            self.era_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_eras),
            )

        if "temporal" in output_tasks:
            # Temporal head (regression for multiple decades)
            self.temporal_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, max(1, len(self.temporal_decades))),
            )

        if "hygiene" in output_tasks:
            self.hygiene_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_hygiene),
            )

        # Language-conditioned ICF refinement: a small MLP that applies a
        # residual correction to the base ICF prediction conditioned on the
        # predicted language distribution.  Only built when both "icf" and
        # "language" tasks are enabled.  At inference the language softmax is
        # detached so gradient flows only through the base and language heads;
        # this keeps the heads independent while still sharing the signal.
        self._use_lang_cond = (
            "icf" in output_tasks and "language" in output_tasks
        )
        if self._use_lang_cond:
            self.lang_icf_cond = nn.Sequential(
                nn.Linear(feature_dim + num_languages, max(feature_dim // 2, 8)),
                nn.ReLU(),
                nn.Linear(max(feature_dim // 2, 8), 1),
                nn.Tanh(),  # Tanh keeps correction in (-1, +1)
            )
            # Scale correction output so it starts small (avoids disrupting warm-start)
            nn.init.zeros_(self.lang_icf_cond[-2].weight)
            nn.init.zeros_(self.lang_icf_cond[-2].bias)

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        Forward pass.

        Args:
            x: [Batch, Max_Char_Len] byte indices
            return_all: If True, return all task outputs; if False, return only ICF
            return_features: If True, also return a feature dict (compatible with UniversalICF)

        Returns:
            If return_all=False: [Batch, 1] ICF predictions
            If return_all=True: Dict with keys: 'icf', 'language', 'era', 'temporal'
        """
        # Base model forward (prefer return_features for compatibility with OOV calibration).
        base_feats: Dict[str, torch.Tensor] = {}
        try:
            icf_pred, base_feats = self.base(x, return_features=True)  # type: ignore[misc]
        except Exception:
            icf_pred = self.base(x)

        # Feature activations: used for auxiliary heads.
        feats = base_feats.get("feature_activations", None)
        if feats is None:
            # Fallback: no feature dict available; best-effort use ICF prediction broadcast.
            feats = icf_pred.detach()

        outputs = {}

        # ICF prediction (from base; always available)
        outputs["icf"] = icf_pred

        # Language prediction
        if "language" in self.output_tasks:
            lang_logits = self.language_head(feats)
            outputs["language"] = lang_logits

            # Language-conditioned ICF refinement: apply a small residual
            # correction to the base ICF scaled by 0.3 so it can nudge within
            # ±0.3 without overriding the base prediction.  The language logits
            # are detached so the correction doesn't feed gradients back into
            # the language head — the two heads remain independently trained.
            if self._use_lang_cond and hasattr(self, "lang_icf_cond"):
                lang_probs = torch.softmax(lang_logits.detach(), dim=-1)
                cond_in = torch.cat([feats.detach(), lang_probs], dim=-1)
                correction = self.lang_icf_cond(cond_in) * 0.3  # scale to ±0.3
                outputs["icf"] = torch.clamp(icf_pred + correction, 0.0, 1.0)

        # Era prediction
        if "era" in self.output_tasks:
            era_logits = self.era_head(feats)
            outputs["era"] = era_logits

        # Temporal prediction
        if "temporal" in self.output_tasks:
            temporal_pred = self.temporal_head(feats)
            temporal_pred = torch.clamp(temporal_pred, 0.0, 1.0)
            outputs["temporal"] = temporal_pred

        # Token hygiene prediction
        if "hygiene" in self.output_tasks:
            hygiene_logits = self.hygiene_head(feats)
            outputs["hygiene"] = hygiene_logits

        if return_features:
            # Preserve base features for OOV calibration. Add any aux logits for convenience.
            out_feats = dict(base_feats)
            for k, v in outputs.items():
                if k != "icf":
                    out_feats[f"{k}_logits"] = v
            return outputs["icf"], out_feats

        if return_all:
            return outputs

        # Return ICF by default (backward compatible).
        return outputs["icf"]
