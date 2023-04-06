import unittest

import torch

from tmodels.graph import (
    GraphTransformer,
    GraphTransformerConfig,
    MultiHeadGraphAttention,
)


class TestMultiHeadGraphAttention(unittest.TestCase):
    def test_attention_forward(self):
        """
        test forward method is going without errors
        keep T1 and T2 different to check if the attention is working properly
        check returned shapes are correct
        """
        model = MultiHeadGraphAttention(
            GraphTransformerConfig(
                d_embed=4,
                n_heads=4,
                attention_dropout=0.1,
                linear_dropout=0.1,
                d_e_embed=5,
                edge_attention_dropout=0.1,
                edge_linear_dropout=0.1,
            )
        )

        nodes1 = torch.randn(2, 3, 4)
        nodes2 = torch.randn(2, 6, 4)
        edges = torch.randn(2, 3, 6, 5)

        new_nodes, new_edges = model(nodes1, edges, x2=nodes2)

        self.assertEqual(new_nodes.shape, (2, 3, 4))
        self.assertEqual(new_edges.shape, (2, 3, 6, 5))

    def test_model_forward(self):
        model = GraphTransformer(
            GraphTransformerConfig(
                bias_embed=True,
                d_node_in=10,
                d_edge_in=20,
                d_node_out=5,
                d_edge_out=7,
                d_embed=4,
                n_heads=4,
                attention_dropout=0.1,
                linear_dropout=0.1,
                d_e_embed=5,
                edge_attention_dropout=0.1,
                edge_linear_dropout=0.1,
            )
        )

        nodes = torch.randn(2, 5, 10)
        edges = torch.randn(2, 5, 5, 20)

        new_nodes, new_edges = model(nodes, edges)

        self.assertEqual(new_nodes.shape, (2, 5, 5))
        self.assertEqual(new_edges.shape, (2, 5, 5, 7))

        nodes = torch.randn(2, 6, 10)
        edges = torch.randn(2, 6, 6, 20)

        new_nodes, new_edges = model(nodes, edges)

        self.assertEqual(new_nodes.shape, (2, 6, 5))
        self.assertEqual(new_edges.shape, (2, 6, 6, 7))

    def test_num_attention_parameters(self):
        model = MultiHeadGraphAttention(
            GraphTransformerConfig(
                d_embed=4,
                n_heads=4,
                d_e_embed=4,
            )
        )

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(params, 888)

    def test_num_model_parameters(self):
        model = GraphTransformer(
            GraphTransformerConfig(
                bias_embed=True,
                d_node_in=10,
                d_edge_in=20,
                d_node_out=5,
                d_edge_out=7,
                d_embed=4,
                n_heads=4,
                d_e_embed=5,
            )
        )

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(params, 15328)
