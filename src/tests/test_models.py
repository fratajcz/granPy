import unittest
from src.nn.encoders import GAE_Encoder
import src.nn.layers as own_layers
import dataclasses
import torch.nn as nn
from src.nn.decoders import DegreeSorter
import torch

class EncoderTest(unittest.TestCase):
    def test_init_nn(self):

        @dataclasses.dataclass
        class opts:
            n_conv_layers = 3
            activation_layer = "ReLU"
            dropout_ratio = 0.5
            mplayer = "GCNConv"
            mplayer_args = []
            mplayer_kwargs = {}
            latent_dim = 32
            layer_ratio = 10

        encoder = GAE_Encoder(1000, opts())

        self.assertTrue(True)

    def test_fancy_activation(self):

        @dataclasses.dataclass
        class opts:
            n_conv_layers = 3
            activation_layer = "Hardswish"
            dropout_ratio = 0.5
            mplayer = "GCNConv"
            mplayer_args = []
            mplayer_kwargs = {}
            latent_dim = 32
            layer_ratio = 10

        encoder = GAE_Encoder(1000, opts())

        self.assertTrue(isinstance(encoder.nn[2], nn.Hardswish))

    def test_mplayer_args(self):

        @dataclasses.dataclass
        class opts:
            n_conv_layers = 3
            activation_layer = "ReLU"
            dropout_ratio = 0.5
            mplayer = "GATConv"
            mplayer_args = [1]
            mplayer_kwargs = {}
            latent_dim = 32
            layer_ratio = 10

        encoder = GAE_Encoder(1000, opts())

        self.assertTrue(encoder.nn[1].heads, opts().mplayer_args[0])

    def test_mplayer_kwargs(self):

        @dataclasses.dataclass
        class opts:
            n_conv_layers = 3
            activation_layer = "ReLU"
            dropout_ratio = 0.5
            mplayer = "GATConv"
            mplayer_args = []
            mplayer_kwargs = {"heads": 1}
            latent_dim = 32
            layer_ratio = 10

        encoder = GAE_Encoder(1000, opts())

        self.assertTrue(encoder.nn[1].heads, opts().mplayer_kwargs["heads"])


    def test_get_own_mplayer(self):

        @dataclasses.dataclass
        class opts:
            n_conv_layers = 3
            activation_layer = "ReLU"
            dropout_ratio = 0.5
            mplayer = "NoneConv"
            mplayer_args = []
            mplayer_kwargs = {}
            latent_dim = 32
            layer_ratio = 10

        encoder = GAE_Encoder(1000, opts())

        self.assertTrue(isinstance(encoder.nn[1], own_layers.NoneConv))

class DegreeSorterTest(unittest.TestCase):
    def test_correct_order(self):
        # node 1,3 and 4 have value 1, node 2 has value 2 and nodes 0,5,6 have value 0
        pos_edges = torch.LongTensor([[0, 0, 1, 1, 1],
                                      [1, 2, 2, 3, 4]])

        neg_edges = torch.LongTensor([[1, 5, 4, 0, 6],
                                      [0, 6, 2, 5, 1]])


        num_nodes = 7

        z = torch.rand((num_nodes, 5))

        sorter = DegreeSorter()
        pos_values = sorter(z, pos_edges, pos_edges)
        self.assertTrue(torch.eq(pos_values, torch.Tensor((1, 2, 2, 1, 1))).all())

        neg_values = sorter(z, neg_edges, pos_edges)
        self.assertTrue(torch.eq(neg_values, torch.Tensor((0, 0, 2, 0, 1))).all())