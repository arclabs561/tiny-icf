"""Nano-CNN model optimized for speed and size (~6.7k parameters)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NanoICF(nn.Module):
    """
    Ultra-compact ICF estimator for Rust deployment.
    
    Architecture optimized for speed:
    - Embedding: 256 -> 16 (fits in L1 cache)
    - Strided Conv1D: stride=2 (50% speedup)
    - Global Max Pool
    - Linear head
    
    Total: ~6,700 parameters (~25KB)
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        emb_dim: int = 16,  # Reduced for speed
        conv_channels: int = 32,
        kernel_size: int = 5,
        stride: int = 2,  # Key speedup: skip every other position
    ):
        super().__init__()
        
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Strided convolution: stride=2 halves the sequence length
        # This is the key speed optimization
        self.conv = nn.Conv1d(
            emb_dim, 
            conv_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=kernel_size // 2,  # Maintain output length
        )
        
        # Simple linear head (no hidden layer for speed)
        self.head = nn.Linear(conv_channels, 1)
    
    def init_weights(self, mean_icf: float = 0.4):
        """Initialize model weights with proper strategies."""
        def init_layer(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.padding_idx is not None:
                    nn.init.constant_(m.weight[m.padding_idx], 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        self.apply(init_layer)
        
        # Initialize final layer bias
        with torch.no_grad():
            if isinstance(self.head, nn.Linear):
                self.head.weight.data *= 0.1
                if self.head.bias is not None:
                    self.head.bias.fill_(mean_icf)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Max_Char_Len] byte indices
        
        Returns:
            [Batch, 1] normalized ICF scores
        """
        # Embed: [Batch, Len] -> [Batch, Len, Emb]
        x_emb = self.emb(x)
        
        # Transpose for Conv1d: [Batch, Emb, Len]
        x_emb = x_emb.transpose(1, 2)
        
        # Strided convolution: [Batch, Emb, Len] -> [Batch, Channels, Len/2]
        conv_out = F.relu(self.conv(x_emb))
        
        # Global Max Pooling: [Batch, Channels, Len/2] -> [Batch, Channels]
        pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
        
        # Linear head + sigmoid
        return torch.sigmoid(self.head(pooled))
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

