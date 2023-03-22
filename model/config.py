class VanillaTransformerConfig:
    def __init__(
        self,
        input_vocab_size=100,
        output_vocab_size=100,
        d_seq=32,
        d_embed=8,
        n_heads=2,
        n_layers=2,
        d_ff=32 * 4,
        emb_dropout=0.1,
        ff_dropout=0.1,
        attention_dropout=0.1,
        linear_dropout=0.1,
        label_smoothing=0.0,
        layer_norm_eps=1**(-5),
        bias=False,
        layer_norm_first=True,
        position_encoding_learned=True,
    ):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.d_seq = d_seq
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.emb_dropout = emb_dropout
        self.ff_dropout = ff_dropout
        self.attention_dropout = attention_dropout
        self.linear_dropout = linear_dropout
        self.label_smoothing = label_smoothing
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        self.layer_norm_first = layer_norm_first
        self.position_encoding_learned = position_encoding_learned
