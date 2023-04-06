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
