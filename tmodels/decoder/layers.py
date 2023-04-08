from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from tmodels.decoder.config import DecoderOnlyConfig
from tmodels.vanilla_transformer.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Feed-forward layer
    """

    def __init__(self, config: DecoderOnlyConfig):
        super().__init__()
        self.feed_forward = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_with_relu",
                        nn.Linear(config.d_embed, config.d_ff, bias=config.bias),
                    ),
                    ("relu", nn.GELU()),
                    (
                        "linear",
                        nn.Linear(config.d_ff, config.d_embed, bias=config.bias),
                    ),
                    ("dropout", nn.Dropout(config.ff_dropout)),
                ]
            )
        )

        self.layer_norm_first = config.layer_norm_first
        self.layer_norm = nn.LayerNorm(config.d_embed, config.layer_norm_eps)

        self.apply(self.init_weights)

    def forward(self, x):
        return self.feed_forward(self.layer_norm(x)) + x

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            # Initialize the weights of the feed-forward layer
            # linear layer followed by ReLU is initialized with gain calculated for ReLU
            torch.nn.init.xavier_uniform_(
                module.linear_with_relu.weight,
                gain=torch.nn.init.calculate_gain("relu"),
            )
            # output linear layer is initialized with default gain=1.
            torch.nn.init.xavier_uniform_(module.linear.weight)


class AttentionLayer(nn.Module):
    """
    Layer block:
    It consists of a layer and a layer normalization.
    Layer normalization is applied before or after the layer.
    It returns attention weights if the layer returns them.
    """

    def __init__(self, config: DecoderOnlyConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm = nn.LayerNorm(config.d_embed, config.layer_norm_eps)

    def forward(self, x, y, mask=None):
        out, _ = self.attention(self.layer_norm(x), self.layer_norm(y), mask=mask)
        out = out + x
        return out


class DecoderLayer(nn.Module):
    """
    Decoder layer:
    It consists of a masked self-attention layer
    Lower triangular mask is used to prevent attention to future tokens in masked self-attention.
    """

    def __init__(self, config: DecoderOnlyConfig):
        super().__init__()
        self.masked_self_attention = AttentionLayer(config)
        self.feed_forward = FeedForward(config)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.d_seq, config.d_seq)).view(
                1, 1, config.d_seq, config.d_seq
            ),
        )

    def forward(self, x):
        mask = self.tril[:, :, : x.size(1), : x.size(1)]
        x = self.masked_self_attention(
            x,
            x,
            mask=mask,
        )
        x = self.feed_forward(x)
        return x


class ModelBlock(nn.Module):
    """
    Model block: It consists of a number of layers.
    It returns list of attention weights if the layers returns them.
    """

    def __init__(self, config: DecoderOnlyConfig, layer):
        super().__init__()
        self.layers = nn.ModuleList([layer(config) for _ in range(config.n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
