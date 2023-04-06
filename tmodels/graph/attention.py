from typing import Optional

import torch
import torch.nn as nn

from tmodels.graph.config import GraphTransformerConfig


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
