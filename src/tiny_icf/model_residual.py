"""Residual ICF Model with residual connections for better gradient flow."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualICF(nn.Module):
    """
    Universal ICF Model with residual connections.
    Based on research showing residual connections are essential for character-level CNNs.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        emb_dim: int = 36,
        conv_channels: int = 18,
        hidden_dim: int = 36,
        dropout: float = 0.4,
    ):
        super().__init__()
        
        # Byte-level embedding
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Parallel Convs with BatchNorm
        self.conv3_block = nn.Sequential(
            nn.Conv1d(emb_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
        )
        self.conv5_block = nn.Sequential(
            nn.Conv1d(emb_dim, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
        )
        self.conv7_block = nn.Sequential(
            nn.Conv1d(emb_dim, conv_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels),
        )
        
        # MLP Head with pre-activation residual connection
        # Research: Pre-activation residuals are better for shallow MLP heads in ranking models
        # Pattern: Norm → Activation → Linear → Residual add (identity path stays clean)
        self.head_bn1 = nn.BatchNorm1d(conv_channels * 9)  # Pre-norm on input
        self.head_linear1 = nn.Linear(conv_channels * 9, hidden_dim)
        self.head_dropout = nn.Dropout(dropout)  # Dropout on residual branch only
        self.head_linear2 = nn.Linear(hidden_dim, 1)
        
        # Projection for residual in head (if dimensions don't match)
        # No activation on projection - identity path stays clean
        self.head_residual_proj = nn.Linear(conv_channels * 9, hidden_dim) if conv_channels * 9 != hidden_dim else nn.Identity()
        
        # Small residual scaling factor for initial stability (learned or fixed)
        # Research: Small initial scale (0.1-0.5) helps head start near identity
        # Use register_buffer to avoid PyTorch Lightning __setattr__ issues
        self.register_buffer('residual_scale', torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed: [Batch, Len] -> [Batch, Len, Emb]
        x_emb = self.emb(x)
        
        # Transpose for Conv1d: [Batch, Emb, Len]
        x_emb = x_emb.transpose(1, 2)
        
        # Extract features
        c3 = F.relu(self.conv3_block(x_emb))
        c5 = F.relu(self.conv5_block(x_emb))
        c7 = F.relu(self.conv7_block(x_emb))
        
        # Multi-scale pooling
        p3_max = F.max_pool1d(c3, c3.size(2)).squeeze(2)
        p3_mean = F.avg_pool1d(c3, c3.size(2)).squeeze(2)
        p3_last = c3[:, :, -1]
        
        p5_max = F.max_pool1d(c5, c5.size(2)).squeeze(2)
        p5_mean = F.avg_pool1d(c5, c5.size(2)).squeeze(2)
        p5_last = c5[:, :, -1]
        
        p7_max = F.max_pool1d(c7, c7.size(2)).squeeze(2)
        p7_mean = F.avg_pool1d(c7, c7.size(2)).squeeze(2)
        p7_last = c7[:, :, -1]
        
        # Concatenate
        combined = torch.cat([
            p3_max, p3_mean, p3_last,
            p5_max, p5_mean, p5_last,
            p7_max, p7_mean, p7_last,
        ], dim=1)  # [Batch, Channels*9]
        
        # Head with pre-activation residual connection
        # Research pattern: Norm → Activation → Linear → Residual add
        # This keeps identity path clean and improves gradient flow
        
        # Pre-norm on input (before activation)
        combined_norm = self.head_bn1(combined)
        combined_act = F.relu(combined_norm)
        
        # Main branch: Linear transformation
        hidden = self.head_linear1(combined_act)
        hidden_dropped = self.head_dropout(hidden)  # Dropout on residual branch only
        
        # Residual connection (project input if needed, no activation on projection)
        residual = self.head_residual_proj(combined)  # Identity path: no norm/activation
        hidden_res = hidden_dropped + (self.residual_scale * residual)  # Pre-activation residual
        
        # Final output layer
        output = self.head_linear2(hidden_res)
        output = torch.clamp(output, 0.0, 1.0)
        
        return output
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_weights(self, mean_icf: float = 0.4):
        """
        Initialize model weights using Kaiming (He) initialization for ReLU.
        
        Research findings:
        - Kaiming initialization is optimal for ReLU activations
        - Final layer should be initialized to predict mean ICF for faster convergence
        - Residual projection should use standard initialization
        """
        # Kaiming initialization for ReLU layers (better than Xavier for ReLU)
        for module in [self.head_linear1, self.head_residual_proj]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
        
        # Initialize final layer to predict mean ICF (conservative initialization)
        nn.init.xavier_uniform_(self.head_linear2.weight, gain=0.1)
        nn.init.constant_(self.head_linear2.bias, mean_icf)
        
        # Initialize conv layers with Kaiming (already done by default in PyTorch, but explicit is better)
        for conv_block in [self.conv3_block, self.conv5_block, self.conv7_block]:
            for layer in conv_block:
                if isinstance(layer, nn.Conv1d):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

