"""
Transformer model.
 Most of the implementation is based on "Attention is all you need" (https://arxiv.org/abs/1706.03762)
"""

import torch
import torch.nn as nn

from model.config import TransformerConfig


class VanilaTransformer(nn.Module):
    """
    Transformer model with:
    - TODO: Embedding layer (vocab size, embedding size)
    - TODO: Positional encoding
    - Encoder
    - Decoder
    - Linear layer for the fincal prediction (vocab size)

    TODO: Add init weights
    TODO: add output dataclass for attention weights
    """
    def __init__(
        self, config: TransformerConfig,
    ):
        super().__init__()
        self.encoder = ModelBlock(config, EncoderLayer)
        self.decoder = ModelBlock(config, DecoderLayer)
        self.linear = nn.Linear(config.d_embed, config.vocab_size)

    def forward(self, x):
        encoded, encoder_attention = self.encoder(x)
        decoded, decoder_attention = self.decoder(encoded)
        output = self.linear(decoded)
        output = torch.softmax(output, dim=1)
        return output, encoder_attention, decoder_attention


class ModelBlock(nn.Module):
    """
    Model block: It consists of a number of layers.
    It returns list of attention weights if the layers returns them.
    """
    def __init__(self, config: TransformerConfig, layer):
        super().__init__()
        self.layers = nn.ModuleList([layer(config) for _ in range(config.n_layers)])

    def forward(self, x):
        attention = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                attention.append(x[1:])
                x = x[0]
        return x, attention


class LayerBlock(nn.Module):
    """
    Layer block:
    It consists of a layer and a layer normalization.
    Layer normalization is applied before or after the layer.
    It returns attention weights if the layer returns them.
    """
    def __init__(self, config: TransformerConfig, model_block):
        super().__init__()
        self.layer = model_block
        self.layer_norm = LayerNorm(config.d_embed, config.layer_norm_eps)
        self.layer_norm_first = config.layer_norm_first if hasattr(config, "layer_norm_first") else False

    def forward(self, x):
        attention = None
        if self.layer_norm_first:
            out = self.layer(self.layer_norm(x))
        else:
            out = self.layer(x)
        if isinstance(out, tuple):
            attention = out[1:]
            out = out[0]
        if self.layer_norm_first:
            out = out + x
        else:
            out = self.layer_norm(out + x)
        if attention is None:
            return out
        return out, attention


class FeedForward(nn.Module):
    """
    Feed-forward layer
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.feed_forward = nn.Sequential()
        self.feed_forward.append(nn.Linear(config.d_embed, config.d_ff, bias=config.bias))
        self.feed_forward.append(nn.ReLU())
        self.feed_forward.append(nn.Linear(config.d_ff, config.d_embed, bias=config.bias))
        self.feed_forward.append(nn.Dropout(config.ff_dropout))

    def forward(self, x):
        return self.feed_forward(x)


class EncoderLayer(nn.Module):
    """
    Encoder layer:
    It consists of a self-attention layer and a feed-forward layer.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = LayerBlock(
            config, MultiHeadAttention(config)
        )
        self.feed_forward = LayerBlock(
            config, FeedForward(config)
        )

    def forward(self, x, mask=None):
        x, self_attention_weights = self.self_attention(x, x, mask=mask)
        x = self.feed_forward(x)
        return x, self_attention_weights


class DecoderLayer(nn.Module):
    """
    Decoder layer:
    It consists of a masked self-attention layer, a cross-attention layer and a feed-forward layer.
    Lower triangular mask is used to prevent attention to future tokens in masked self-attention.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.masked_self_attention = LayerBlock(
            config, MultiHeadAttention(config)
        )
        self.register_buffer("tril", torch.tril(torch.ones(config.d_seq, config.d_seq))
                             .view(1, 1, config.d_seq, config.d_seq))

        self.cross_attention = LayerBlock(
            config, MultiHeadAttention(config)
        )
        self.feed_forward = LayerBlock(
            config, FeedForward(config)
        )

    def forward(self, x, y, self_mask=None, cross_mask=None):
        """
        Args:
            x: query
            y: key and value
            self_mask: mask for masked self-attention
            cross_mask: mask for cross-attention
        Returns: output, (masked_attention_weights, cross_attention_weights)
        """
        if self_mask is None:
            mask = self.tril[:, :, :x.size(1), :x.size(1)]
        else:
            mask = self_mask.view(1, 1, self_mask.size(0), self_mask.size(1)) & self.tril[:, :, :x.size(1), :x.size(1)]

        x, masked_attention_weights = self.masked_self_attention(x, x, mask=mask)
        x, cross_attention_weights = self.cross_attention(x, y, mask=cross_mask)
        x = self.feed_forward(x)
        return x, (masked_attention_weights, cross_attention_weights)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer
    It supports both self-attention and cross-attention.
    Masking could be used for self-attention to prevent attention to future tokens,
     or cross-attention to prevent attention to padding tokens,
     or any other arbitrary masking.
    Forward method returns attention weights for visualization.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_embed % config.n_heads == 0
        self.values = nn.Linear(config.d_embed, config.d_embed * config.n_heads)
        self.keys = nn.Linear(config.d_embed, config.d_embed * config.n_heads)
        self.queries = nn.Linear(config.d_embed, config.d_embed * config.n_heads)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.n_heads = config.n_heads
        self.linear = nn.Linear(config.n_heads * config.d_embed, config.d_embed)
        self.linear_dropout = nn.Dropout(config.linear_dropout)

    def forward(self, q, k, v=None, mask=None):
        assert q.shape == k.shape
        if v is not None:
            assert k.shape == v.shape
        B, T, C = q.shape  # (B = batch size , T = sequence length, C = embedding dim)
        assert C % self.n_heads == 0

        q = self.queries(q)
        k = self.keys(k)
        if v in None:
            v = self.values(k)
        else:
            v = self.values(v)

        # (B, T, n_heads * hidden_size) -> (B, T, n_heads, hidden_size) -> (B, n_heads, T, hidden_size)
        # where n_heads * hidden_size = C
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # (B, n_heads, T, hidden_size) * (B, n_heads, hidden_size, T) = (B, n_heads, T, T)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(k.shape[-1])
        if mask is not None:
            scaled_qk = scaled_qk.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scaled_qk, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # (B, n_heads, T, T) * (B, n_heads, T, hidden_size) = (B, n_heads, T, hidden_size)
        output = torch.matmul(attention_weights, v)

        # (B, n_heads, T, hidden_size) ->  (B, T, n_heads, hidden_size) -> (B, T, n_heads * hidden_size)
        # (B, T, n_heads * hidden_size) = (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.linear(output)
        output = self.linear_dropout(output)

        return output, attention_weights


class LayerNorm(nn.Module):
    """
    Layer Normalization from scratch (https://arxiv.org/abs/1607.06450)
    It is done by computing the mean and variance for all
    inputs to the neurons in a layer on a single sample in batch independently.
    """
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean)**2).mean(-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.weight + self.bias
        return y
