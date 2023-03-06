class TransformerConfig:
    def __init__(
        self,
        vocab_size,
        d_seq,
        d_embed,
        n_heads,
        n_layers,
        d_ff,
        dropout,
        max_len,
        label_smoothing=0.0,
        layer_norm_eps=1e-12,
        bias=False,
        layer_norm_first=False
    ):
        self.vocab_size = vocab_size
        self.d_seq = d_seq
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_len = max_len
        self.label_smoothing = label_smoothing
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        self.layer_norm_first = layer_norm_first
