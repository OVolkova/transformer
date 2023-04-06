import torch
import torch.nn as nn
from torch.autograd import Variable

from tmodels.vanila_transformer.config import VanillaTransformerConfig


class Embeddings(nn.Module):
    """
    Embedding layer for the input and output.
    It consists of:
    - Embedding layer
    - Positional encoding
    - Dropout layer
    """

    def __init__(self, vocab_size, config: VanillaTransformerConfig):
        super().__init__()
        self.scaling_dims = torch.sqrt(torch.tensor(config.d_embed))
        # Embeddings are just a lookup table of size (vocab_size, embedding_size).
        # For each token, we get a vector of size embedding_size from the lookup table.
        # The embedding layer could be considered as a linear layer without bias, just weights.
        # It is initialized with random weights and trained along with the model.
        self.embedding = nn.Embedding(vocab_size, config.d_embed)

        # Positional encoding in the original paper could be done 2 ways:
        if config.position_encoding_learned:
            #  1. Learned encoding
            self.positional_encoding = nn.Embedding(config.d_seq, config.d_embed)
        else:
            # 2. Sinusoid encoding
            self.positional_encoding = SinusoidPositionalEncoding(
                config.d_seq, config.d_embed
            )

        # dropout layer is applied to the sum of embeddings and positional encoding
        self.dropout = nn.Dropout(config.emb_dropout)

        self.apply(self.init_weights)

    def forward(self, x):
        embedded = self.embedding(x) * self.scaling_dims  # (batch_size, seq_len, d_embed)
        pos = torch.arange(0, embedded.size(1)).unsqueeze(0)
        positions = self.positional_encoding(pos)
        return self.dropout(embedded + positions)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)


class SinusoidPositionalEncoding(nn.Module):
    """
    Sinusoid positional encoding.
    It is used to add positional information to the input.
    It is a fixed function of the position.
    i.e. the same position will always have the same encoding and is not learnable.
    It is using the formula:
        PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    The implementation is done with exponentiation and log to avoid division and power of big numbers.
    It is based on the fact that:
        1/M^N = exp(log(1/M^N)) = exp(-N*log(M))
    Where:
        M = 10000, N = 2i/d_model
    """

    def __init__(self, d_seq, d_embed):
        super().__init__()
        self.d_embed = d_embed
        self.d_seq = d_seq
        pe = torch.zeros(d_seq, d_embed)
        wavelengths = torch.exp(
            torch.arange(0, d_embed, 2) / d_embed * -torch.log(torch.tensor(10000.0))
        )
        pe[:, 0::2] = torch.sin(torch.arange(0, d_seq).unsqueeze(1) * wavelengths)
        pe[:, 1::2] = torch.cos(torch.arange(0, d_seq).unsqueeze(1) * wavelengths)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return Variable(self.pe[: x.size(1), :], requires_grad=False)
