import unittest
from src.nn.encoders import GNNEncoder, MLPEncoder
import dataclasses
import torch.nn as nn
from src.nn.decoders import DegreeSorter, MLPDecoder, CorrelationDecoder
from src.nn.models import NaiveModel
from scipy.stats import pearsonr
import src.nn.models as models
import torch

class NaiveModelTest(unittest.TestCase):
    def test_get_model(self):
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
            decoder = "HarmonicDegreeSorter"
            model = "NaiveModel"

        model = getattr(models, opts().model)(100, opts())

        self.assertTrue(isinstance(model, NaiveModel))

    def test_init(self):

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
            decoder = "HarmonicDegreeSorter"

        model = NaiveModel(100, opts())

        self.assertTrue(True)

    def test_encode(self):
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
            decoder = "HarmonicDegreeSorter"

        model = NaiveModel(100, opts())

        x = torch.rand((5, 10))

        out = model.encode(x)

        out2 = model.encode(x, x)

        self.assertTrue(torch.equal(x, out))
        self.assertTrue(torch.equal(x, out2))

        self.assertTrue(True)

    def test_decode(self):
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
            decoder = "HarmonicDegreeSorter"

        model = NaiveModel(100, opts())

        x = torch.rand((5, 10))
        edges = torch.LongTensor([[0, 1, 2, 4],
                                  [1, 2, 3, 3]])
        
        prediction = model.decode(x, edges, edges)

        self.assertTrue(torch.equal(prediction, torch.Tensor([1, 1, 4/3, 4/3])))

        

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

        encoder = GNNEncoder(1000, opts())

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

        encoder = GNNEncoder(1000, opts())

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

        encoder = GNNEncoder(1000, opts())

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

        encoder = GNNEncoder(1000, opts())

        self.assertTrue(encoder.nn[1].heads, opts().mplayer_kwargs["heads"])

    def test_MLPEncoder(self):
        @dataclasses.dataclass
        class opts:
            n_conv_layers = 2
            activation_layer = "ReLU"
            dropout_ratio = 0.5
            mplayer_args = []
            mplayer_kwargs = {}
            latent_dim = 32
            layer_ratio = 10

        encoder = MLPEncoder(100, opts())

        x = torch.rand(5, 100)
        edge_index = torch.LongTensor([[0, 1, 3],
                                       [3, 2, 1]])
        
        z = encoder(x, edge_index)


class DecoderTest(unittest.TestCase):
    def test_forward(self):
        @dataclasses.dataclass
        class opts:
            n_conv_layers = 2
            activation_layer = "ReLU"
            dropout_ratio = 0.5
            mplayer_args = []
            mplayer_kwargs = {}
            latent_dim = 32
            layer_ratio = 10

        decoder = MLPDecoder(opts())

        x = torch.rand((5, 32))
        edge_index = torch.LongTensor([[0, 1, 3],
                                       [3, 2, 1]])
        
        output = decoder(x, edge_index)
        self.assertEqual(output.shape[0], 3)

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
        
class CorrelationDecoderTest(unittest.TestCase):
    def test_correct_values(self):
        # node 1,3 and 4 have value 1, node 2 has value 2 and nodes 0,5,6 have value 0
        pos_edges = torch.LongTensor([[0, 0, 1, 1, 1],
                                      [1, 2, 2, 3, 4]])

        num_nodes = 7

        z = torch.rand((num_nodes, 5))
        
        @dataclasses.dataclass
        class opts:
            n_conv_layers = 2
            activation_layer = "ReLU"
            dropout_ratio = 0.5
            mplayer_args = []
            mplayer_kwargs = {}
            latent_dim = 32
            layer_ratio = 10

        correlation = CorrelationDecoder(opts)
        out_values = correlation(z, pos_edges, sigmoid=False)
        true_values = torch.tensor([pearsonr(z[edge[0]],  z[edge[1]]).statistic for edge in pos_edges.T], dtype=torch.float32)
        self.assertTrue(torch.allclose(true_values, out_values))