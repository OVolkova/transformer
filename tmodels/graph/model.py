from collections import OrderedDict

import torch
import torch.nn as nn

from tmodels.graph.attention import MultiHeadGraphAttention
from tmodels.graph.config import GraphTransformerConfig


class GraphAttentionLayer(nn.Module):
    """
    Layer block:
    It consists of a layer and a layer normalization.
    Layer normalization is applied before or after the layer.
    It returns attention weights if the layer returns them.
    """

    def __init__(self, config: GraphTransformerConfig):
        super().__init__()
        self.attention = MultiHeadGraphAttention(config)
        self.layer_norm_x = nn.LayerNorm(config.d_embed, config.layer_norm_eps)
        self.layer_norm_e = nn.LayerNorm(config.d_e_embed, config.layer_norm_eps)

    def forward(self, x, e):
        out_x, out_e = self.attention(x1=self.layer_norm_x(x), e=self.layer_norm_e(e))
        out_x = out_x + x
        out_e = out_e + e
        return out_x, out_e


class FeedForward(nn.Module):
    """
    Feed-forward layer
    """

    def __init__(self, d_embed, d_ff, has_bias=True, dropout=0.0, layer_norm_eps=1e-5):
        super().__init__()
        self.feed_forward = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_with_relu",
                        nn.Linear(d_embed, d_ff, bias=has_bias),
                    ),
                    ("relu", nn.GELU()),
                    (
                        "linear",
                        nn.Linear(d_ff, d_embed, bias=has_bias),
                    ),
                    ("dropout", nn.Dropout(dropout)),
                ]
            )
        )
        self.layer_norm = nn.LayerNorm(d_embed, layer_norm_eps)
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


class ModelBlock(nn.Module):
    """
    Model block: It consists of a number of layers.
    It returns list of attention weights if the layers returns them.
    """

    def __init__(self, config: GraphTransformerConfig, layer):
        super().__init__()
        self.layers = nn.ModuleList([layer(config) for _ in range(config.n_layers)])

    def forward(self, x, e):
        for layer in self.layers:
            x, e = layer(x, e)
        return x, e


class GraphTransformerLayer(nn.Module):
    """
    Graph model layer:
    It consists of a self-attention layer, and a feed-forward layer for nodes features and for edge features.
    """

    def __init__(self, config: GraphTransformerConfig):
        super().__init__()
        self.self_attention = GraphAttentionLayer(config)
        self.feed_forward_x = FeedForward(
            config.d_embed,
            config.d_ff,
            config.bias,
            config.ff_dropout,
            config.layer_norm_eps,
        )
        self.feed_forward_e = FeedForward(
            config.d_e_embed,
            config.d_e_ff,
            config.bias,
            config.e_ff_dropout,
            config.layer_norm_eps,
        )

    def forward(self, x, e):
        x, e = self.self_attention(x, e)
        x = self.feed_forward_x(x)
        e = self.feed_forward_e(e)
        return x, e


class GraphTransformer(nn.Module):
    """
    Transformer model for Graphs
    """

    def __init__(
        self,
        config: GraphTransformerConfig,
    ):
        super().__init__()
        self.node_embedding = nn.Linear(
            config.d_node_in, config.d_embed, bias=config.bias_embed
        )
        self.edge_embedding = nn.Linear(
            config.d_edge_in, config.d_e_embed, bias=config.bias_embed
        )
        self.encoder = ModelBlock(config, GraphTransformerLayer)

        self.node_linear = nn.Linear(config.d_embed, config.d_node_out)
        self.edge_linear = nn.Linear(config.d_e_embed, config.d_edge_out)

    def forward(self, x, e):
        x = self.node_embedding(x)
        e = self.edge_embedding(e)
        x, e = self.encoder(x, e)
        x = self.node_linear(x)
        e = self.edge_linear(e)
        return x, e


if __name__ == "__main__":
    config_ = GraphTransformerConfig()
    model = GraphTransformer(config_)
    print(model)
