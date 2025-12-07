"""Universal ICF Model: Byte-level CNN for word frequency estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class UniversalICF(nn.Module):
    """
    Universal Frequency Estimator using byte-level CNNs.
    
    Architecture:
    - Byte embedding (256 -> 64)
    - Parallel 1D CNNs (kernel sizes 3, 5, 7) for morphological patterns
    - Global max pooling
    - MLP head with sigmoid output (0.0 = common, 1.0 = rare)
    
    Total parameters: < 50k
    """
    
    def __init__(
        self,
        vocab_size: int = 256,  # UTF-8 bytes
        emb_dim: int = 36,  # Further reduced from 48 to reduce capacity
        conv_channels: int = 18,  # Further reduced from 24 to reduce capacity
        hidden_dim: int = 36,  # Further reduced from 48 to reduce capacity
        dropout: float = 0.4,  # Increased from 0.3 for stronger regularization
        use_attention: bool = False,  # Enable multi-head self-attention
        attention_heads: int = 4,  # Number of attention heads
    ):
        super().__init__()
        
        # Byte-level embedding
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Parallel Convs: capture n-grams of different sizes
        # Kernel 3: trigrams like "ing", "pre"
        # Kernel 5: roots like "graph"
        # Kernel 7: complex affixes
        # Added BatchNorm for better generalization
        self.conv3 = nn.Sequential(
            nn.Conv1d(emb_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(emb_dim, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(emb_dim, conv_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels),
        )
        
        # Multi-head self-attention (optional, for long-range dependencies)
        self.use_attention = use_attention
        if use_attention:
            # Attention operates on concatenated conv outputs
            # Input: [batch, seq_len, conv_channels*3] (after concatenating c3, c5, c7)
            attention_dim = conv_channels * 3
            self.attention: Optional[nn.MultiheadAttention] = nn.MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            # After attention, we still do multi-scale pooling
            # So input to head remains conv_channels * 9
        else:
            self.attention = None  # type: ignore
        
        # MLP Head
        # Use clipped linear instead of sigmoid to avoid saturation
        # Sigmoid saturates at extremes, causing gradient vanishing
        # Input size is now conv_channels * 9 (multi-scale pooling)
        self.head = nn.Sequential(
            nn.Linear(conv_channels * 9, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            # No sigmoid - we'll clip in forward pass
        )
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Args:
            x: [Batch, Max_Char_Len] byte indices
            return_features: If True, return additional features and metadata
        
        Returns:
            If return_features=False: [Batch, 1] normalized ICF scores
            If return_features=True: (scores, features_dict) where features_dict contains:
                - 'icf_score': [Batch, 1] normalized ICF scores
                - 'raw_output': [Batch, 1] raw model output before clamping
                - 'feature_activations': [Batch, hidden_dim] hidden layer activations
                - 'confidence': [Batch, 1] confidence estimate (based on feature magnitude)
        """
        # Embed: [Batch, Len] -> [Batch, Len, Emb]
        x_emb = self.emb(x)
        
        # Transpose for Conv1d: [Batch, Emb, Len]
        x_emb = x_emb.transpose(1, 2)
        
        # Extract morphological features (BatchNorm is inside Sequential now)
        c3 = F.relu(self.conv3(x_emb))  # [Batch, Channels, SeqLen]
        c5 = F.relu(self.conv5(x_emb))  # [Batch, Channels, SeqLen]
        c7 = F.relu(self.conv7(x_emb))  # [Batch, Channels, SeqLen]
        
        # Apply attention if enabled (for long-range dependencies)
        if self.use_attention and self.attention is not None:
            # Concatenate conv outputs: [Batch, Channels*3, SeqLen]
            conv_concat = torch.cat([c3, c5, c7], dim=1)  # [Batch, Channels*3, SeqLen]
            # Transpose for attention: [Batch, SeqLen, Channels*3]
            conv_concat = conv_concat.transpose(1, 2)
            # Apply self-attention
            attended, _ = self.attention(conv_concat, conv_concat, conv_concat)
            # Transpose back: [Batch, Channels*3, SeqLen]
            attended = attended.transpose(1, 2)
            # Split back into c3, c5, c7 for pooling
            split_size = c3.size(1)
            c3 = attended[:, :split_size, :]
            c5 = attended[:, split_size:2*split_size, :]
            c7 = attended[:, 2*split_size:, :]
        
        # Multi-scale pooling: Max + Mean + Last token
        # Global max pooling loses positional information
        # Multi-scale captures more information than max alone
        p3_max = F.max_pool1d(c3, c3.size(2)).squeeze(2)  # [Batch, Channels]
        p3_mean = F.avg_pool1d(c3, c3.size(2)).squeeze(2)  # Mean pooling
        p3_last = c3[:, :, -1]  # Last position
        
        p5_max = F.max_pool1d(c5, c5.size(2)).squeeze(2)
        p5_mean = F.avg_pool1d(c5, c5.size(2)).squeeze(2)
        p5_last = c5[:, :, -1]
        
        p7_max = F.max_pool1d(c7, c7.size(2)).squeeze(2)
        p7_mean = F.avg_pool1d(c7, c7.size(2)).squeeze(2)
        p7_last = c7[:, :, -1]
        
        # Concatenate multi-scale features (3 scales × 3 kernels = 9 feature sets)
        combined = torch.cat([
            p3_max, p3_mean, p3_last,
            p5_max, p5_mean, p5_last,
            p7_max, p7_mean, p7_last,
        ], dim=1)  # [Batch, Channels*9]
        
        # Pass through head layers (extract intermediate features for return_features)
        if return_features:
            # Extract features from intermediate layers
            # head[0]: Linear -> head[1]: BatchNorm -> head[2]: ReLU -> head[3]: Dropout -> head[4]: Linear
            hidden = self.head[0](combined)  # [Batch, hidden_dim] - Linear
            hidden_bn = self.head[1](hidden)  # [Batch, hidden_dim] - BatchNorm
            feature_activations = self.head[2](hidden_bn)  # [Batch, hidden_dim] - ReLU
            hidden_dropped = self.head[3](feature_activations)  # [Batch, hidden_dim] - Dropout
            raw_output = self.head[4](hidden_dropped)  # Final Linear: [Batch, 1]
            output = torch.clamp(raw_output, 0.0, 1.0)  # [Batch, 1]
        else:
            # Standard forward pass (backward compatible)
            output = self.head(combined)  # [Batch, 1]
            output = torch.clamp(output, 0.0, 1.0)  # [Batch, 1]
            raw_output = output
            feature_activations = None
        
        if return_features:
            # Compute confidence estimate (based on feature activation magnitude)
            # Higher activation magnitude = more confident prediction
            if feature_activations is not None:
                feature_magnitude = torch.norm(feature_activations, dim=1, keepdim=True)  # [Batch, 1]
                # Normalize to [0, 1] range (rough estimate)
                confidence = torch.sigmoid(feature_magnitude / feature_activations.size(1))
            else:
                # Fallback: use output magnitude as confidence proxy
                confidence = torch.sigmoid(torch.abs(raw_output) * 10.0)
            
            # Use feature_activations if available, otherwise use a zero tensor as fallback
            features_dict = {
                'icf_score': output.clone(),
                'raw_output': raw_output.clone(),
                'confidence': confidence.clone(),
            }
            if feature_activations is not None:
                features_dict['feature_activations'] = feature_activations.clone()
            else:
                # Create a zero tensor with same shape as hidden would have been
                hidden_dim = self.head[0].out_features if hasattr(self.head[0], 'out_features') else hidden_dim
                features_dict['feature_activations'] = torch.zeros(
                    (output.size(0), hidden_dim),
                    device=output.device, dtype=output.dtype
                )
            
            return output.clone(), features_dict
        
        return output  # [Batch, 1]
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_weights(self, mean_icf: float = 0.4):
        """
        Initialize model weights with proper strategies.
        
        Args:
            mean_icf: Expected mean ICF value (for final layer bias initialization)
        """
        def init_layer(m):
            if isinstance(m, nn.Embedding):
                # Embeddings: small normal initialization
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.padding_idx is not None:
                    nn.init.constant_(m.weight[m.padding_idx], 0.0)
            elif isinstance(m, nn.Linear):
                # Linear layers: Kaiming normal for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                # Conv layers: Kaiming normal for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Apply initialization
        self.apply(init_layer)
        
        # Initialize final layer bias to approximate mean ICF
        # This helps model start near expected output range
        # Also scale final layer weights smaller to prevent saturation
        with torch.no_grad():
            if hasattr(self.head, '__getitem__'):
                # Sequential module
                final_layer = self.head[-1]
                if isinstance(final_layer, nn.Linear):
                    # Scale weights smaller to prevent initial saturation
                    final_layer.weight.data *= 0.1
                    if final_layer.bias is not None:
                        final_layer.bias.fill_(mean_icf)

