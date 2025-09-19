'''
=================================================
coding:utf-8
@Time:      2025/8/14 21:26
@File:      TMSANet.py
@Author:    Ziwei Wang
@Function:
=================================================
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# Multi-scale 1D Convolution Module
class MultiScaleConv1d(nn.Module):
    """
    Multi-scale 1D convolution module to extract features with multiple kernel sizes.
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels for each convolution.
        kernel_sizes: List of kernel sizes for each convolution layer.
        padding: List of padding values for each kernel size.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes, padding):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=p) for k, p in zip(kernel_sizes, padding)
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))  # Batch normalization after concatenation
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        # Apply each convolution and concatenate the results along the channel dimension
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)  # Concatenate along channel axis
        out = self.bn(out)  # Apply batch normalization
        out = self.dropout(out)  # Apply dropout
        return out


# Multi-Headed Attention Module with Local and Global Attention
class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention mechanism combining local and global attention.
    Args:
        d_model: Dimensionality of input features.
        n_head: Number of attention heads.
        dropout: Dropout rate for regularization.
    """
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head  # Dimensionality per attention head for keys
        self.d_v = d_model // n_head  # Dimensionality per attention head for values
        self.n_head = n_head

        # Multi-scale convolution settings for local feature extraction
        kernel_sizes = [3, 5]
        padding = [1, 2]

        self.multi_scale_conv_k = MultiScaleConv1d(d_model, d_model, kernel_sizes, padding)

        # Linear projections for queries, local keys, global keys, and values
        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k_local = nn.Linear(d_model * len(kernel_sizes), n_head * self.d_k)
        self.w_k_global = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_v)
        self.w_o = nn.Linear(n_head * self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Forward pass for local and global attention combination.
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model).
            key: Key tensor of shape (batch_size, seq_len, d_model).
            value: Value tensor of shape (batch_size, seq_len, d_model).
        """
        bsz = query.size(0)

        # Local key extraction using multi-scale convolution
        key_local = key.transpose(1, 2)  # Transpose to (batch_size, d_model, seq_len)
        key_local = self.multi_scale_conv_k(key_local).transpose(1, 2)

        # Linear projections
        q = self.w_q(query).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)  # Query
        k_local = self.w_k_local(key_local).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)  # Local Key
        k_global = self.w_k_global(key).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)  # Global Key
        v = self.w_v(value).view(bsz, -1, self.n_head, self.d_v).transpose(1, 2)  # Value

        # Local attention
        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_local = F.softmax(scores_local, dim=-1)
        attn_local = self.dropout(attn_local)
        x_local = torch.matmul(attn_local, v)

        # Global attention
        scores_global = torch.matmul(q, k_global.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_global = F.softmax(scores_global, dim=-1)
        attn_global = self.dropout(attn_global)
        x_global = torch.matmul(attn_global, v)

        # Combine local and global attention outputs
        x = x_local + x_global

        # Concatenate results and project to output dimensions
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.n_head * self.d_v)
        return self.w_o(x)


# Feed-Forward Neural Network
class FeedForward(nn.Module):
    """
    Two-layer feed-forward network with GELU activation.
    Args:
        d_model: Dimensionality of input and output features.
        d_hidden: Dimensionality of the hidden layer.
        dropout: Dropout rate for regularization.
    """
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()  # Activation function
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)  # Linear layer 1
        x = self.act(x)  # Activation
        x = self.dropout(x)  # Dropout
        x = self.w_2(x)  # Linear layer 2
        x = self.dropout(x)  # Dropout
        return x


# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    """
    A single transformer encoder layer with multi-head attention and feed-forward network.
    Args:
        embed_dim: Dimensionality of input embeddings.
        num_heads: Number of attention heads.
        fc_ratio: Ratio for expanding the feed-forward hidden layer.
        attn_drop: Dropout rate for attention mechanism.
        fc_drop: Dropout rate for feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)  # LayerNorm after attention
        self.layernorm2 = nn.LayerNorm(embed_dim)  # LayerNorm after feed-forward

    def forward(self, data):
        # Apply attention with residual connection and layer normalization
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)

        # Apply feed-forward network with residual connection and layer normalization
        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output

# Feature Extraction Module
class ExtractFeature(nn.Module):
    """
    Extracts temporal and spatial features from input data using convolutional layers.
    Args:
        num_channels: Number of input channels (e.g., sensors or features).
        num_samples: Number of time points in the input sequence.
        embed_dim: Output dimensionality of the embedding.
        pool_size: Kernel size for average pooling.
        pool_stride: Stride size for average pooling.
    """
    def __init__(self, num_channels, num_samples, embed_dim, pool_size, pool_stride):
        super().__init__()
        # Temporal convolution with different kernel sizes
        self.temp_conv1 = nn.Conv2d(1, embed_dim, (1, 31), padding=(0, 15))
        self.temp_conv2 = nn.Conv2d(1, embed_dim, (1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(embed_dim)  # Batch normalization for temporal features

        # Spatial convolution across all channels
        self.spatial_conv1 = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1), padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(embed_dim)  # Batch normalization for spatial features
        self.glu = nn.GELU()  # Activation function
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)  # Temporal average pooling

    def forward(self, x):
        """
        Forward pass for feature extraction.
        Args:
            x: Input tensor of shape (batch_size, num_channels, num_samples).
        Returns:
            Output tensor with extracted features.
        """
        x = x.unsqueeze(dim=1)  # Add a channel dimension -> (batch_size, 1, num_channels, num_samples)
        x1 = self.temp_conv1(x)  # Temporal convolution with kernel size 31
        x2 = self.temp_conv2(x)  # Temporal convolution with kernel size 15
        x = x1 + x2  # Combine features from both convolutions
        x = self.bn1(x)  # Apply batch normalization
        x = self.spatial_conv1(x)  # Spatial convolution
        x = self.glu(x)  # Apply activation function
        x = self.bn2(x)  # Apply batch normalization
        x = x.squeeze(dim=2)  # Remove spatial dimension -> (batch_size, embed_dim, num_samples)
        x = self.avg_pool(x)  # Apply average pooling
        return x


# Transformer Module
class TransformerModule(nn.Module):
    """
    Stacks multiple transformer encoder layers.
    Args:
        embed_dim: Dimensionality of input embeddings.
        num_heads: Number of attention heads in each encoder layer.
        fc_ratio: Expansion ratio for feed-forward layers.
        depth: Number of transformer encoder layers.
        attn_drop: Dropout rate for attention mechanism.
        fc_drop: Dropout rate for feed-forward layers.
    """
    def __init__(self, embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop):
        super().__init__()
        # Create a list of transformer encoder layers
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)
        ])

    def forward(self, x):
        """
        Forward pass for the transformer module.
        Args:
            x: Input tensor of shape (batch_size, embed_dim, num_samples).
        Returns:
            Transformed tensor with the same shape.
        """
        x = rearrange(x, 'b d n -> b n d')  # Rearrange to (batch_size, seq_len, embed_dim)
        for encoder in self.transformer_encoders:
            x = encoder(x)  # Pass through each encoder layer
        x = x.transpose(1, 2)  # Rearrange back to (batch_size, embed_dim, seq_len)
        x = x.unsqueeze(dim=2)  # Add a spatial dimension -> (batch_size, embed_dim, 1, seq_len)
        return x


# Classification Module
class ClassifyModule(nn.Module):
    """
    Performs classification based on extracted features.
    Args:
        embed_dim: Dimensionality of embeddings.
        temp_embedding_dim: Dimensionality of temporal embeddings after pooling.
        num_classes: Number of output classes.
    """
    def __init__(self, embed_dim, temp_embedding_dim, num_classes):
        super().__init__()
        # Fully connected layer for classification
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        """
        Forward pass for classification.
        Args:
            x: Input tensor of shape (batch_size, embed_dim, 1, seq_len).
        Returns:
            Classification logits of shape (batch_size, num_classes).
        """
        x = x.reshape(x.size(0), -1)  # Flatten the input tensor
        out = self.classify(x)  # Pass through the classification layer
        return out


# Complete TMSA-Net Model
class TMSANet(nn.Module):
    """
    TMSA-Net: Combines feature extraction, transformer encoders, and classification modules.
    Args:
        in_planes (int): Number of input channels (e.g., sensors).
        radix (int): Radix factor, typically set to 1.
        time_points (int): Number of time points in the input sequence.
        num_classes (int): Number of output classes for classification.
        embed_dim (int): Dimensionality of embeddings.
            - Use 19 for BCIC-IV-2a.
            - Use 6 for BCIC-IV-2b.
            - Use 10 for HGD.
        pool_size (int): Kernel size for pooling.
        pool_stride (int): Stride size for pooling.
        num_heads (int): Number of attention heads in the transformer.
        fc_ratio (int): Expansion ratio for feed-forward layers.
        depth (int): Depth of the transformer encoder (number of layers).
        attn_drop (float): Dropout rate for attention mechanism.
            - Set to 0.7 for HGD dataset.
        fc_drop (float): Dropout rate for feed-forward layers.
    """
    def __init__(self, in_planes, radix, time_points, num_classes, embed_dim=19, pool_size=50,
                 pool_stride=15, num_heads=4, fc_ratio=2, depth=1, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.in_planes = in_planes * radix  # Adjust input dimensionality
        self.extract_feature = ExtractFeature(self.in_planes, time_points, embed_dim, pool_size, pool_stride)
        temp_embedding_dim = (time_points - pool_size) // pool_stride + 1  # Compute temporal embedding size
        self.dropout = nn.Dropout()  # Dropout layer before transformer
        self.transformer_module = TransformerModule(embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop)
        self.classify_module = ClassifyModule(embed_dim, temp_embedding_dim, num_classes)

    def forward(self, x):
        """
        Forward pass for TMSA-Net.
        Args:
            x: Input tensor of shape (batch_size, in_planes, time_points).
        Returns:
            Classification logits of shape (batch_size, num_classes).
        """
        x = self.extract_feature(x)  # Extract features
        x = self.dropout(x)  # Apply dropout
        feas = self.transformer_module(x)  # Apply transformer module
        out = self.classify_module(feas)  # Classify the features
        return feas, out


# Main function to test the model
if __name__ == '__main__':
    # Instantiate the model
    block = TMSANet(22, 1, 1000, 4)

    # Generate random input data (batch_size=16, channels=22, time_points=1000)
    input = torch.rand(16, 22, 1000)

    # Perform forward pass
    output = block(input)

    # Calculate total number of trainable parameters
    total_trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')  # Print the total parameters

    # Print model architecture
    print(block)