import unittest
from src.nn.encoders import GAE_Encoder
import src.nn.layers as own_layers
import dataclasses
import torch.nn as nn


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

