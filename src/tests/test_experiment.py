import dataclasses
import unittest
from src.experiment import Experiment
import shutil
import os

class ExperimentTest(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree("src/tests/data/processed", ignore_errors=True)
        shutil.rmtree("src/tests/models", ignore_errors=True)

    def test_init(self):

        @dataclasses.dataclass
        class opts:
            n_conv_layers = 1
            activation_layer = "ReLU"
            dropout_ratio = 0.5
            mplayer = "GCNConv"
            mplayer_args = []
            mplayer_kwargs = {}
            latent_dim = 32
            layer_ratio = 10
            root = "src/tests/data/"
            dataset = "mccallatest"
            model_path = "src/test/models/"
            lr = 1e-3
            es_patience = 5
            decoder = "InnerProductDecoder"

        experiment = Experiment(opts())

    def test_run(self):

        @dataclasses.dataclass
        class opts:
            n_conv_layers = 1
            activation_layer = "ReLU"
            dropout_ratio = 0.5
            mplayer = "GCNConv"
            mplayer_args = []
            mplayer_kwargs = {}
            latent_dim = 32
            layer_ratio = 10
            root = "src/tests/data/"
            dataset = "mccallatest"
            model_path = "src/test/models/"
            lr = 1e-3
            es_patience = 5
            decoder = "InnerProductDecoder"
            epochs = 10

        _opts = opts()
        experiment = Experiment(_opts)

        experiment.run()

        self.assertIsNotNone(experiment.test_performance)  # if it has a non-None test performance
        dataset_path = os.path.join(_opts.root, "processed", experiment.hash + ".pt")
        self.assertTrue(os.path.isfile(dataset_path))  # if it has produced its dataset
        self.assertTrue(os.path.isfile(os.path.join(_opts.model_path, experiment.hash + ".pt")))  # if it has produced a model file