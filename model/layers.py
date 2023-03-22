from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.config import VanillaTransformerConfig


class LayerNorm(nn.Module):
    """
    Layer Normalization from scratch (https://arxiv.org/abs/1607.06450)
    It is done by computing the mean and variance for all
    inputs to the neurons in a layer on a single sample in batch independently.
    Gamma and Beta are trainable parameters, initialized to 1 and 0 respectively.
    """

    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.gamma + self.beta
        return y


class FeedForward(nn.Module):
    """
    Feed-forward layer
    """

    def __init__(self, config: VanillaTransformerConfig):
        super().__init__()
        self.feed_forward = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_with_relu",
                        nn.Linear(config.d_embed, config.d_ff, bias=config.bias),
                    ),
                    ("relu", nn.ReLU()),
                    (
                        "linear",
                        nn.Linear(config.d_ff, config.d_embed, bias=config.bias),
                    ),
                    ("dropout", nn.Dropout(config.ff_dropout)),
                ]
            )
        )

        self.layer_norm_first = config.layer_norm_first
        self.layer_norm = LayerNorm(config.d_embed, config.layer_norm_eps)

        self.apply(self.init_weights)

    def forward(self, x):
        if self.layer_norm_first:
            out = self.feed_forward(self.layer_norm(x))
            out = out + x
        else:
            out = self.feed_forward(x)
            out = self.layer_norm(out + x)
        return out

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


@dataclass
class AttentionLayerOutput:
    self_attention: torch.Tensor
    cross_attention: Optional[torch.Tensor]


@dataclass
class AttentionOutput:
    encoder: List[AttentionLayerOutput]
    decoder: List[AttentionLayerOutput]


class AttentionLayer(nn.Module):
    """
    Layer block:
    It consists of a layer and a layer normalization.
    Layer normalization is applied before or after the layer.
    It returns attention weights if the layer returns them.
    """

    def __init__(self, config: VanillaTransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm = LayerNorm(config.d_embed, config.layer_norm_eps)
        self.layer_norm_first = config.layer_norm_first

    def forward(self, x, y, mask=None):
        if self.layer_norm_first:
            out, attention = self.attention(
                self.layer_norm(x), self.layer_norm(y), mask=mask
            )
            out = out + x
        else:
            out, attention = self.attention(x, y, mask=mask)
            out = self.layer_norm(out + x)
        return out, attention


class ModelBlock(nn.Module):
    """
    Model block: It consists of a number of layers.
    It returns list of attention weights if the layers returns them.
    """

    def __init__(self, config: VanillaTransformerConfig, layer):
        super().__init__()
        self.layers = nn.ModuleList([layer(config) for _ in range(config.n_layers)])

    def forward(self, x, y=None, mask=None):
        attention = []
        for layer in self.layers:
            x = layer(x, y=y, mask=mask)
            attention.append(x[1:])
            x = x[0]
        return x, attention


class EncoderLayer(nn.Module):
    """
    Encoder layer:
    It consists of a self-attention layer and a feed-forward layer.
    """

    def __init__(self, config: VanillaTransformerConfig):
        super().__init__()
        self.self_attention = AttentionLayer(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, y=None, mask=None):
        mask = mask.view(mask.size(0), 1, 1, mask.size(1)) if mask is not None else None
        x, self_attention_weights = self.self_attention(
            x,
            x,
            mask=mask,
        )
        x = self.feed_forward(x)
        return x, AttentionLayerOutput(
            self_attention=self_attention_weights, cross_attention=None
        )


class DecoderLayer(nn.Module):
    """
    Decoder layer:
    It consists of a masked self-attention layer, a cross-attention layer and a feed-forward layer.
    Lower triangular mask is used to prevent attention to future tokens in masked self-attention.
    """

    def __init__(self, config: VanillaTransformerConfig):
        super().__init__()
        self.masked_self_attention = AttentionLayer(config)
        self.cross_attention = AttentionLayer(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, y, mask=None):
        mask = mask.view(mask.size(0), 1, 1, mask.size(1)) if mask is not None else None
        x, masked_attention_weights = self.masked_self_attention(
            x,
            x,
            mask=mask,
        )
        x, cross_attention_weights = self.cross_attention(x, y)
        x = self.feed_forward(x)
        return x, AttentionLayerOutput(
            self_attention=masked_attention_weights,
            cross_attention=cross_attention_weights,
        )
