"""
Transformer model.
 Most of the implementation is based on "Attention is all you need" (https://arxiv.org/abs/1706.03762)
"""

import torch
import torch.nn as nn

from model.config import TransformerConfig
from model.embeddings import Embeddings
from model.layers import AttentionOutput, DecoderLayer, EncoderLayer, ModelBlock


class VanillaTransformer(nn.Module):
    """
    Transformer model with Encoder and Decoder.
    Both Input and Output are transformed to embeddings with positional encoding.
    The next token prediction is done by a linear layer on the output of the decoder.

    TODO: softmax on output
    TODO: add generation method
    """

    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__()
        self.encoder_embedding = Embeddings(config.input_vocab_size, config)
        self.encoder = ModelBlock(config, EncoderLayer)

        self.decoder_embedding = Embeddings(config.output_vocab_size, config)
        self.decoder = ModelBlock(config, DecoderLayer)

        self.linear = nn.Linear(config.d_embed, config.output_vocab_size)

    def forward(self, x, y, encoder_mask=None, decoder_mask=None):
        encoded, encoder_attention = self.encode(x, encoder_mask)
        output, decoder_attention = self.decode(y, encoded, decoder_mask)
        return output, AttentionOutput(
            encoder=encoder_attention, decoder=decoder_attention
        )

    def encode(self, x, mask=None):
        embedded = self.encoder_embedding(x)
        encoded, attention = self.encoder(embedded, mask=mask)
        return encoded, attention

    def decode(self, x, encoded, mask=None):
        embedded = self.decoder_embedding(x)
        decoded, attention, cross_attention = self.decoder(
            embedded, y=encoded, mask=mask
        )
        output = self.linear(decoded)
        output = torch.softmax(output, dim=1)
        return output, attention


if __name__ == "__main__":
    config_ = TransformerConfig()
    model = VanillaTransformer(config_)
