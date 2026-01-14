# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
# ]
# ///
"""Create a model variant with residual connections based on research findings."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Parallel Convs with residual connections
        # Each conv block: Conv -> BatchNorm -> ReLU -> (residual connection)
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
        
        # Projection layers for residual connections (if needed)
        # For now, we'll use simple addition where dimensions match
        
        # MLP Head with residual connection
        self.head_linear1 = nn.Linear(conv_channels * 9, hidden_dim)
        self.head_bn1 = nn.BatchNorm1d(hidden_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.head_linear2 = nn.Linear(hidden_dim, 1)
        
        # Projection for residual in head (if dimensions don't match)
        self.head_residual_proj = nn.Linear(conv_channels * 9, hidden_dim) if conv_channels * 9 != hidden_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed: [Batch, Len] -> [Batch, Len, Emb]
        x_emb = self.emb(x)
        
        # Transpose for Conv1d: [Batch, Emb, Len]
        x_emb = x_emb.transpose(1, 2)
        
        # Extract features with residual-like connections
        # Note: True residual requires same input/output dimensions
        # For different kernel sizes, we use the conv outputs directly
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
        
        # Head with residual connection
        hidden = self.head_linear1(combined)
        hidden_bn = self.head_bn1(hidden)
        hidden_act = F.relu(hidden_bn)
        
        # Residual connection (project input if needed)
        residual = self.head_residual_proj(combined)
        hidden_res = hidden_act + residual  # Residual connection
        
        hidden_dropped = self.head_dropout(hidden_res)
        output = self.head_linear2(hidden_dropped)
        output = torch.clamp(output, 0.0, 1.0)
        
        return output
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_weights(self, mean_icf: float = 0.4):
        """Initialize model weights."""
        # Initialize final layer to predict mean ICF
        nn.init.xavier_uniform_(self.head_linear2.weight, gain=0.1)
        nn.init.constant_(self.head_linear2.bias, mean_icf)

if __name__ == "__main__":
    # Test the model
    model = ResidualICF(emb_dim=36, conv_channels=18, hidden_dim=36, dropout=0.4)
    print(f"ResidualICF parameters: {model.count_parameters():,}")
    
    dummy_input = torch.randint(0, 256, (4, 10))
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("✓ ResidualICF model works correctly")

