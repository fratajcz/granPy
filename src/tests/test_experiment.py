import dataclasses
import unittest
from src.experiment import Experiment
import shutil
import os
from torch_geometric.data import Data
import torch
from typing import List

@dataclasses.dataclass
class experimentTestOpts:
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
    val_mode = "max"
    val_metric = "average_precision_score"
    test_metrics = ["average_precision_score", "roc_auc_score"]
    val_seed = 2
    canonical_test_seed = 1
    score_batched = False
    test_fraction = 0.2
    val_fraction = 0.2
    cuda = False
    epochs = 5
    negative_sampling = "unstructure"
    test_metrics: List[str] = dataclasses.field(default_factory=list)


class ExperimentTest(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree("src/tests/data/processed", ignore_errors=True)
        shutil.rmtree("src/tests/models", ignore_errors=True)

    def test_init(self):
        experiment = Experiment(experimentTestOpts())

    def test_run(self):

        _opts = experimentTestOpts()
        experiment = Experiment(_opts)

        experiment.run()

        self.assertIsNotNone(experiment.test_performance)  # if it has a non-None test performance
        dataset_path = os.path.join(_opts.root, "processed", experiment.hash + ".pt")
        self.assertTrue(os.path.isfile(dataset_path))  # if it has produced its dataset
        self.assertTrue(os.path.isfile(os.path.join(_opts.model_path, experiment.hash + ".pt")))  # if it has produced a model file

    def test_run_score_batched(self):

        _opts = experimentTestOpts()
        _opts.score_batched = True
        experiment = Experiment(_opts)

        experiment.run()

        self.assertIsNotNone(experiment.test_performance)  # if it has a non-None test performance
        dataset_path = os.path.join(_opts.root, "processed", experiment.hash + ".pt")
        self.assertTrue(os.path.isfile(dataset_path))  # if it has produced its dataset
        self.assertTrue(os.path.isfile(os.path.join(_opts.model_path, experiment.hash + ".pt")))  # if it has produced a model file

    def test_compare_performance(self):

        _opts = experimentTestOpts()
        experiment = Experiment(_opts)

        is_better = experiment.is_best_val_performance(experiment.best_val_performance - 1)

        self.assertFalse(is_better)

        is_better = experiment.is_best_val_performance(experiment.best_val_performance + 1)

        self.assertTrue(is_better)
        
    def test_patience(self):
        _opts = experimentTestOpts()
        experiment = Experiment(_opts)

        self.assertEqual(experiment.patience, _opts.es_patience)

        isoutofpatience = experiment.out_of_patience(False)

        self.assertFalse(isoutofpatience)
        self.assertEqual(experiment.patience, _opts.es_patience - 1)

        for _ in range(experiment.patience - 1):
            # exhaust patience
            isoutofpatience = experiment.out_of_patience(False)
            self.assertFalse(isoutofpatience)

        # one more tick exhausts it
        isoutofpatience = experiment.out_of_patience(False)
        self.assertTrue(isoutofpatience)

        # restore patience
        isoutofpatience = experiment.out_of_patience(True)
        self.assertFalse(isoutofpatience)
        self.assertEqual(experiment.patience, _opts.es_patience)

    def test_batch_pos_edges_by_tf(self):
        data = Data(edge_index=torch.LongTensor([[0, 1, 3, 4],
                                                 [3, 4, 0, 1]]))
        
        pot_net = [None, None, torch.LongTensor([0, 1])]

        directed_pos_edges, batch_mask = Experiment.batch_pos_edges_by_tf(data, pot_net)

        self.assertTrue(torch.equal(directed_pos_edges, torch.LongTensor([[0, 1],
                                                                          [3, 4]])))
        self.assertTrue(torch.equal(batch_mask, torch.LongTensor([0, 1])))
