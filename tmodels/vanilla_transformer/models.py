"""
Transformer model.
 Most of the implementation is based on "Attention is all you need" (https://arxiv.org/abs/1706.03762)
"""

import torch
import torch.nn as nn

from tmodels.vanilla_transformer.config import VanillaTransformerConfig
from tmodels.vanilla_transformer.embeddings import Embeddings
from tmodels.vanilla_transformer.layers import (
    AttentionOutput,
    DecoderLayer,
    EncoderLayer,
    ModelBlock,
)


class VanillaTransformer(nn.Module):
    """
    Transformer model with Encoder and Decoder.
    Both Input and Output are transformed to embeddings with positional encoding.
    The next token prediction is done by a linear layer on the output of the decoder.
    """

    def __init__(
        self,
        config: VanillaTransformerConfig,
    ):
        super().__init__()
        self.encoder_embedding = Embeddings(config.input_vocab_size, config)
        self.encoder = ModelBlock(config, EncoderLayer)

        self.decoder_embedding = Embeddings(config.output_vocab_size, config)
        self.decoder = ModelBlock(config, DecoderLayer)

        self.linear = nn.Linear(config.d_embed, config.output_vocab_size)

    def forward(self, x, targets, encoder_mask=None, decoder_mask=None):
        encoded, encoder_attention = self.encode(x, encoder_mask)
        output, decoder_attention = self.decode(targets, encoded, decoder_mask)
        return output, AttentionOutput(
            encoder=encoder_attention, decoder=decoder_attention
        )

    def encode(self, x, mask=None):
        embedded = self.encoder_embedding(x)
        encoded, attention = self.encoder(embedded, mask=mask)
        return encoded, attention

    def decode(self, x, encoded, mask=None):
        embedded = self.decoder_embedding(x)
        decoded, attention = self.decoder(embedded, y=encoded, mask=mask)
        logits = self.linear(decoded)
        return logits, attention

    def generate(
        self, x, max_len, do_sample=False, top_k=None, sos_token_id=0, encoder_mask=None
    ):
        targets = torch.Tensor([sos_token_id]).long().unsqueeze(0).repeat(x.shape[0], 1)
        for _ in range(max_len):
            logits, _ = self(
                x, targets=targets, encoder_mask=encoder_mask, decoder_mask=None
            )
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs[:, -1, :], num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            targets = torch.cat([targets, idx_next], dim=-1)

        return targets


if __name__ == "__main__":
    config_ = VanillaTransformerConfig()
    model = VanillaTransformer(config_)
    print(model)
