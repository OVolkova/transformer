from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


class GraphTransformerConfig:
    def __init__(
        self,
        d_embed=8,
        n_heads=4,
        n_layers=4,
        d_ff=32 * 4,
        ff_dropout=0.1,
        attention_dropout=0.1,
        linear_dropout=0.1,
        layer_norm_eps=1,
        bias=True,
        # graph specific part of the config starts here
        d_e_embed=8,
        d_e_ff=32 * 4,
        edge_attention_dropout=0.1,
        edge_linear_dropout=0.1,
        e_ff_dropout=0.1,
        bias_embed=True,
        d_node_in=10,
        d_edge_in=20,
        d_node_out=10,
        d_edge_out=20,
    ):
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.ff_dropout = ff_dropout
        self.attention_dropout = attention_dropout
        self.linear_dropout = linear_dropout
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias

        # graph specific part of the config starts here
        self.d_e_embed = d_e_embed
        self.edge_attention_dropout = edge_attention_dropout
        self.edge_linear_dropout = edge_linear_dropout
        self.d_e_ff = d_e_ff
        self.e_ff_dropout = e_ff_dropout
        self.bias_embed = bias_embed
        self.d_node_in = d_node_in
        self.d_edge_in = d_edge_in
        self.d_node_out = d_node_out
        self.d_edge_out = d_edge_out


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-Head Attention Graph Layer
    It is a graph layer that uses multi-head attention
     to compute the new node embeddings and edges embeddings.

    The node embeddings are computed as follows:
    1. compute  scaled_QK = Q*K/sqrt(d_k) for each head
    2. compute edge embeddings with attention for edges features as follows:
        2.0. Q = E, K = scaled_QK, V = E
            dimension of Q, K, V: (B, n_head, T1, T2, E)
        2.1. fixing T1, compute attention and output as usual
        2.2. fixing T2, compute attention and output as usual
        2.3. sum the results of 2.1 and 2.2
        2.4. apply linear layer and dropout
    3. compute attention for node features as follows:
        3.1. fixing T1, scaled_EQK1 = scaled_QK * edge embeddings/sqrt(d_e)
        3.2. fixing T2, scaled_EQK2 = scaled_QK * edge embeddings/sqrt(d_e)
        3.3. compute softmax for scaled_EQK1 and scaled_EQK2
    4. compute outputs as usual for scaled_EQK1 and scaled_EQK2
    5. sum the results of 3.4
    6. apply linear layer and dropout

    The result will be new node embeddings and new edges embeddings.
    """

    def __init__(self, config: GraphTransformerConfig):
        super().__init__()
        assert config.d_embed % config.n_heads == 0
        self.n_heads = config.n_heads
        self.values = nn.Linear(config.d_embed, config.d_embed * config.n_heads)
        self.keys = nn.Linear(config.d_embed, config.d_embed * config.n_heads)
        self.queries = nn.Linear(config.d_embed, config.d_embed * config.n_heads)

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.linear = nn.Linear(config.n_heads * config.d_embed, config.d_embed)
        self.linear_dropout = nn.Dropout(config.linear_dropout)

        # edges features
        self.edge_queries = nn.Linear(config.d_e_embed, config.d_e_embed * config.n_heads)
        self.edge_values = nn.Linear(config.d_e_embed, config.d_e_embed * config.n_heads)
        self.edge_nodes_keys = nn.Linear(
            config.n_heads, config.d_e_embed * config.n_heads
        )

        self.edge_attention_dropout = nn.Dropout(config.edge_attention_dropout)
        self.edge_linear = nn.Linear(
            config.n_heads * config.d_e_embed, config.n_heads * config.d_e_embed
        )
        self.edge_linear_embed = nn.Linear(
            config.n_heads * config.d_e_embed, config.d_e_embed
        )
        self.edge_linear_dropout = nn.Dropout(config.edge_linear_dropout)

        self.apply(self.init_weights)

    def edge_attention(
        self, e: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # e: (B, T, T, E')
        q = self.edge_queries(e)  # (B, T, T, E') -> (B, T, T, E)
        v = self.edge_values(e)
        k = self.edge_nodes_keys(
            k.transpose(-3, -1).transpose(-3, -2)
        )  # (B, n_head, T, T) -> (B, T, T, E)

        B, T2, T1, E = q.shape
        # (B = batch size , T = sequence length, E = edges embedding dim)
        assert E % self.n_heads == 0
        # assert T1 == T

        # (B, T2, T1, n_heads * hidden_size) -> (B, T2, T1, n_heads, hidden_size) -> (B, n_heads, T1, T2, hidden_size)
        # where n_heads * hidden_size = E
        k = k.view(B, T2, T1, self.n_heads, E // self.n_heads).transpose(-4, -2)
        q = q.view(B, T2, T1, self.n_heads, E // self.n_heads).transpose(-4, -2)
        v = v.view(B, T2, T1, self.n_heads, E // self.n_heads).transpose(-4, -2)

        # (B, n_heads, T1, T2, hidden_size) * (B, n_heads, T1, hidden_size, T2) = (B, n_heads, T1, T2, T2)
        scaled_qk1 = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(k.size(-1))
        )
        # (B, n_heads, T2, T1, hidden_size) * (B, n_heads, T2, hidden_size, T1) = (B, n_heads, T2, T1, T1)
        scaled_qk2 = torch.matmul(
            q.transpose(-3, -2), k.transpose(-3, -2).transpose(-2, -1)
        ) / torch.sqrt(torch.tensor(k.size(-1)))
        attention_weights1 = torch.softmax(scaled_qk1, dim=-1)
        attention_weights1 = self.edge_attention_dropout(attention_weights1)
        attention_weights2 = torch.softmax(scaled_qk2, dim=-1)
        attention_weights2 = self.edge_attention_dropout(attention_weights2)

        # (B, n_heads, T1, T2, T2) * (B, n_heads, T1, T2, hidden_size) = (B, n_heads, T1, T2, hidden_size)
        output1 = torch.matmul(attention_weights1, v)
        # (B, n_heads, T2, T1, T1) * (B, n_heads, T2, T1, hidden_size) = (B, n_heads, T1, T2, hidden_size)
        output2 = torch.matmul(attention_weights2, v.transpose(-3, -2)).transpose(-3, -2)
        output = output1 + output2

        # (B, n_heads, T1, T2, hidden_size) ->  (B, T2, T1, n_heads, hidden_size) -> (B, T2, T1, n_heads * hidden_size)
        # (B, T2, T1, n_heads * hidden_size) = (B, T, T, E)
        output = output.transpose(1, 3).contiguous().view(B, T2, T1, E)
        output = self.edge_linear(output)
        output = self.edge_linear_dropout(output)
        embeddings = self.edge_linear_embed(output)
        embeddings = self.edge_linear_dropout(embeddings)
        output = output.view(B, T2, T1, self.n_heads, E // self.n_heads).transpose(
            -4, -2
        )  # (B, n_heads, T1, T2, E // self.n_heads)

        return output, embeddings

    def forward(
        self, x1: torch.Tensor, e: torch.Tensor, x2: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.queries(x1)
        v = self.values(x2 if x2 is not None else x1)
        k = self.keys(x2 if x2 is not None else x1)

        B, T2, C = q.shape  # (B = batch size , T = sequence length, C = embedding dim)
        assert C % self.n_heads == 0
        T1 = k.shape[1]

        # (B, T, n_heads * hidden_size) -> (B, T, n_heads, hidden_size) -> (B, n_heads, T, hidden_size)
        # where n_heads * hidden_size = C
        k = k.view(B, T1, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T2, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T1, self.n_heads, C // self.n_heads).transpose(1, 2)

        # (B, n_heads, T, hidden_size) * (B, n_heads, hidden_size, T) = (B, n_heads, T2, T1)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(k.size(-1))
        )
        # edges features
        e, edge_embedding = self.edge_attention(
            e, scaled_qk
        )  # (B, n_heads, T1, T2, E // n_heads)
        # (B = batch size , T = sequence length, E = edges embedding dim)

        # (B, n_heads, T2, 1, T1) * (B, n_heads, T2, T1, e_hidden_size) = (B, n_heads, T2, 1, e_hidden_size)
        scaled_eqk2 = torch.matmul(
            scaled_qk.unsqueeze(-2), e.transpose(-3, -2)
        ) / torch.sqrt(torch.tensor(T1))
        # (B, n_heads, T1, 1, T2) * (B, n_heads, T1, T2, e_hidden_size) = (B, n_heads, T1, 1, e_hidden_size)
        scaled_eqk1 = torch.matmul(
            scaled_qk.transpose(-2, -1).unsqueeze(-2), e
        ) / torch.sqrt(torch.tensor(T2))

        attention_weights_e1 = torch.softmax(scaled_eqk1, dim=-1)
        attention_weights_e1 = self.attention_dropout(attention_weights_e1)
        attention_weights_e2 = torch.softmax(scaled_eqk2, dim=-1)
        attention_weights_e2 = self.attention_dropout(attention_weights_e2)

        # (B, n_heads, T1, 1, e_hidden_size) * (B, n_heads, T1, e_hidden_size, T2) = (B, n_heads, T1, 1, T2)
        # (B, n_heads, T1, 1, T2) -> (B, n_heads, T1, T2)> (B, n_heads, T1, T2)
        output_e1 = (
            torch.matmul(attention_weights_e1, e.transpose(-2, -1))
            .squeeze(-2)
            .transpose(-2, -1)
        )
        # (B, n_heads, T2, 1, e_hidden_size) * (B, n_heads, T2, e_hidden_size, T1) = (B, n_heads, T2, 1, T1)
        # (B, n_heads, T2, 1, T1) -> (B, n_heads, T2, T1) -> (B, n_heads, T1, T2)
        output_e2 = torch.matmul(
            attention_weights_e2, e.transpose(-3, -2).transpose(-2, -1)
        ).squeeze(-2)

        # (B, n_heads, T2, T1) * (B, n_heads, T1, hidden_size) = (B, n_heads, T2, hidden_size)
        output = torch.matmul(output_e1, v) + torch.matmul(output_e2, v)

        # (B, n_heads, T, hidden_size) ->  (B, T, n_heads, hidden_size) -> (B, T, n_heads * hidden_size)
        # (B, T, n_heads * hidden_size) = (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T2, C)
        output = self.linear(output)
        output = self.linear_dropout(output)

        return output, edge_embedding

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


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
