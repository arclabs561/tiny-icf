"""Unified weight initialization utilities for all model variants."""

import torch.nn as nn


def init_weights_xavier(module: nn.Module, gain: float = 1.0) -> None:
    """Initialize weights using Xavier uniform initialization."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def init_weights_kaiming(module: nn.Module, mode: str = "fan_in") -> None:
    """Initialize weights using Kaiming (He) initialization."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def init_embedding(embedding: nn.Embedding, std: float = 0.1) -> None:
    """Initialize embedding layer with small random values."""
    nn.init.normal_(embedding.weight, mean=0.0, std=std)
    if embedding.padding_idx is not None:
        embedding.weight.data[embedding.padding_idx].fill_(0.0)


def init_final_layer(
    linear: nn.Linear,
    mean_target: float = 0.4,
    weight_scale: float = 0.1,
) -> None:
    """
    Initialize final output layer to prevent initial saturation.

    Args:
        linear: Final linear layer
        mean_target: Expected mean of output (e.g., mean ICF)
        weight_scale: Scale factor for weights (smaller = less aggressive)
    """
    # Scale down weights to prevent saturation
    nn.init.xavier_uniform_(linear.weight, gain=weight_scale)
    # Initialize bias to target mean
    nn.init.constant_(linear.bias, mean_target)


def init_model_weights(
    model: nn.Module,
    mean_icf: float = 0.4,
    final_layer_scale: float = 0.1,
    embedding_std: float = 0.1,
) -> None:
    """
    Initialize all model weights with consistent strategy.

    Args:
        model: Model to initialize
        mean_icf: Expected mean ICF value (for final layer bias)
        final_layer_scale: Weight scale for final layer
        embedding_std: Standard deviation for embedding initialization
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            init_embedding(module, std=embedding_std)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            # Check if this is the final output layer
            if "head" in name.lower() and len(list(module.children())) == 0:
                # Try to find the actual final linear layer
                # This is a heuristic - may need adjustment per model
                if hasattr(module, "weight") and module.out_features == 1:
                    init_final_layer(module, mean_icf, final_layer_scale)
                else:
                    init_weights_xavier(module)
            else:
                init_weights_xavier(module)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            # BatchNorm/LayerNorm: standard initialization
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
