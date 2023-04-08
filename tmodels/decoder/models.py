"""
Decoder Only model
"""

import torch
import torch.nn as nn

from tmodels.decoder.config import DecoderOnlyConfig
from tmodels.decoder.layers import DecoderLayer, ModelBlock
from tmodels.vanilla_transformer.embeddings import Embeddings


class DecoderOnly(nn.Module):
    """
    Transformer model with Encoder and Decoder.
    Both Input and Output are transformed to embeddings with positional encoding.
    The next token prediction is done by a linear layer on the output of the decoder.
    """

    def __init__(
        self,
        config: DecoderOnlyConfig,
    ):
        super().__init__()
        self.d_seq = config.d_seq
        self.embedding = Embeddings(config.vocab_size, config)
        self.transformer = ModelBlock(config, DecoderLayer)

        self.linear = nn.Linear(config.d_embed, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        logits = self.linear(x)
        return logits

    def generate(self, x, max_len, do_sample=False, top_k=None, last_token=None):
        init_len = x.shape[-1]
        shifted = False
        for _ in range(max_len):
            logits = self(x)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs[:, -1, :], num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            x = torch.cat([x, idx_next], dim=-1)
            if shifted or idx_next == last_token:
                break
            if not shifted and x.shape[-1] > self.d_seq:
                x = x[:, 1:]
                shifted = True
        return x if shifted else x[:, 1:]


if __name__ == "__main__":
    config_ = DecoderOnlyConfig()
    model = DecoderOnly(config_)
    print(model)
