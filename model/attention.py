import torch
import torch.nn as nn

from model.config import TransformerConfig


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

        self.apply(self.init_weights)

    def forward(self, q, k, mask=None):
        assert q.shape == k.shape
        B, T, C = q.shape  # (B = batch size , T = sequence length, C = embedding dim)
        assert C % self.n_heads == 0

        q = self.queries(q)
        k = self.keys(k)
        v = self.values(k)

        # (B, T, n_heads * hidden_size) -> (B, T, n_heads, hidden_size) -> (B, n_heads, T, hidden_size)
        # where n_heads * hidden_size = C
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # (B, n_heads, T, hidden_size) * (B, n_heads, hidden_size, T) = (B, n_heads, T, T)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(k.shape[-1])
        if mask is not None:
            scaled_qk = scaled_qk.masked_fill(mask == 0, float("-inf"))
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

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
