"""Transformer-based ICF model using HuggingFace architectures.

Supports multiple transformer architectures:
- Character-level: ByT5, Charformer
- Token-level: DistilBERT, TinyBERT, MobileBERT, BERT-base
- Efficient: DeBERTa-v3, RoBERTa-base
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoConfig,
        ByT5Model,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, transformer models disabled")


class TransformerICF(nn.Module):
    """
    Transformer-based ICF model using HuggingFace architectures.

    Supports multiple architectures:
    - Character-level: ByT5, Charformer
    - Token-level: DistilBERT, TinyBERT, MobileBERT, BERT-base
    - Efficient: DeBERTa-v3, RoBERTa-base

    Architecture:
    - HuggingFace transformer backbone (frozen or fine-tuned)
    - Pooling layer (mean, max, cls, or learned)
    - MLP head for ICF prediction

    Note: Token-level models (BERT, DistilBERT, RoBERTa) require word strings
    for tokenization. For byte-level input, use CharacterLevelTransformerICF
    with ByT5 instead.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        pooling: Literal["mean", "max", "cls", "learned"] = "mean",
        hidden_dim: int = 128,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        use_pretrained: bool = True,
        max_length: int = 128,  # Max sequence length for tokenization
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for TransformerICF")

        self.model_name = model_name
        self.pooling = pooling
        self.freeze_backbone = freeze_backbone
        self.max_length = max_length

        # Load tokenizer for token-level models
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"⚠️  Could not load tokenizer: {e}")
            self.tokenizer = None

        # Load transformer model
        try:
            if use_pretrained:
                self.backbone = AutoModel.from_pretrained(model_name)
            else:
                config = AutoConfig.from_pretrained(model_name)
                self.backbone = AutoModel.from_config(config)

            # Get embedding dimension
            if hasattr(self.backbone.config, "hidden_size"):
                self.embedding_dim = self.backbone.config.hidden_size
            elif hasattr(self.backbone.config, "d_model"):
                self.embedding_dim = self.backbone.config.d_model
            else:
                # Fallback: try to infer from first layer
                self.embedding_dim = 768  # Default BERT size

            # Freeze backbone if requested
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print(f"✅ Frozen transformer backbone: {model_name}")
            else:
                print(f"✅ Fine-tuning transformer backbone: {model_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        # Pooling layer
        if pooling == "learned":
            self.pooling_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        else:
            self.pooling_layer = None

        # MLP head for ICF prediction
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        word_strings: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input byte tensor [batch, seq_len] (0-255 byte values)
               OR token IDs [batch, seq_len] if already tokenized
            attention_mask: Optional attention mask [batch, seq_len]
            word_strings: Optional list of word strings for tokenization
                         (required for token-level models like BERT/DistilBERT)

        Returns:
            ICF predictions [batch, 1]
        """
        # Token-level models (BERT, DistilBERT, RoBERTa) need word strings
        # If word_strings provided, tokenize them
        if word_strings is not None and self.tokenizer is not None:
            # Tokenize word strings
            tokenized = self.tokenizer(
                word_strings,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(x.device)
            attention_mask = tokenized["attention_mask"].to(x.device)
        elif x.dtype == torch.long and x.max() < 50000:  # Likely token IDs
            # Assume x is already token IDs
            input_ids = x
            if attention_mask is None:
                # Create attention mask (non-zero = valid token)
                attention_mask = (input_ids != 0).long()
        else:
            # Byte-level input: convert bytes to word strings, then tokenize
            # This is a fallback - ideally use CharacterLevelTransformerICF for byte input
            if self.tokenizer is not None:
                # Convert bytes to strings (simple approach: decode as UTF-8)
                word_strings = []
                for byte_seq in x:
                    try:
                        word = bytes(byte_seq.cpu().numpy().astype(np.uint8)).decode(
                            "utf-8", errors="ignore"
                        )
                        word_strings.append(word)
                    except Exception:
                        word_strings.append("")

                tokenized = self.tokenizer(
                    word_strings,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = tokenized["input_ids"].to(x.device)
                attention_mask = tokenized["attention_mask"].to(x.device)
            else:
                # No tokenizer: use bytes directly (may not work for token-level models)
                input_ids = x.long()
                if attention_mask is None:
                    attention_mask = (input_ids != 0).long()

        # Get transformer outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract hidden states
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        elif hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]  # Last layer
        else:
            # Some models return tuple
            hidden_states = outputs[0]  # [batch, seq_len, hidden_dim]

        # Pooling
        if self.pooling == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)
        elif self.pooling == "max":
            if attention_mask is not None:
                # Masked max pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                hidden_states = hidden_states.masked_fill(mask_expanded == 0, float("-inf"))
                pooled = hidden_states.max(dim=1)[0]
                pooled = pooled.masked_fill(torch.isinf(pooled), 0.0)
            else:
                pooled = hidden_states.max(dim=1)[0]
        elif self.pooling == "cls":
            # Use first token (CLS token for BERT, first token for others)
            pooled = hidden_states[:, 0, :]
        elif self.pooling == "learned":
            # Learned pooling (attention-weighted)
            if self.pooling_layer is not None:
                # Simple learned pooling: linear transformation + mean
                pooled = self.pooling_layer(hidden_states).mean(dim=1)
            else:
                pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states.mean(dim=1)

        # MLP head
        output = self.head(pooled)
        output = torch.clamp(output, 0.0, 1.0)

        return output


class CharacterLevelTransformerICF(nn.Module):
    """
    Character-level transformer model for ICF prediction.

    Uses character-level tokenization with transformer backbone.
    Supports ByT5, Charformer, or character-level BERT.
    """

    def __init__(
        self,
        model_name: str = "google/byt5-small",
        hidden_dim: int = 128,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for CharacterLevelTransformerICF")

        self.model_name = model_name

        # Load character-level transformer
        try:
            if "byt5" in model_name.lower():
                self.backbone = ByT5Model.from_pretrained(model_name)
                self.embedding_dim = self.backbone.config.d_model
            else:
                # Fallback to AutoModel
                self.backbone = AutoModel.from_pretrained(model_name)
                if hasattr(self.backbone.config, "hidden_size"):
                    self.embedding_dim = self.backbone.config.hidden_size
                elif hasattr(self.backbone.config, "d_model"):
                    self.embedding_dim = self.backbone.config.d_model
                else:
                    self.embedding_dim = 512  # Default for character models

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print(f"✅ Frozen character-level transformer: {model_name}")
            else:
                print(f"✅ Fine-tuning character-level transformer: {model_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to load character-level model {model_name}: {e}")

        # Pooling and head
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with character-level input."""
        outputs = self.backbone(input_ids=x, attention_mask=attention_mask)

        # Extract hidden states
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Mean pooling (character-level models typically use mean)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)

        # MLP head
        output = self.head(pooled)
        output = torch.clamp(output, 0.0, 1.0)

        return output
