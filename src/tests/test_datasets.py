import unittest
from src.datasets.datasets import McCallaDataset, GranPyDataset, DatasetBootstrapper
from torch_geometric.data import Data
import pathlib
import shutil
import os
import requests
import torch
import dataclasses

class McCallaDatasetDownloadTest(unittest.TestCase):
    def setUp(self):
        pathlib.Path("test_data").mkdir(parents=True, exist_ok=True)    

    def tearDown(self):
        shutil.rmtree("test_data", ignore_errors=True)

    def test_ping_edgelist(self):
        response = requests.head(McCallaDataset.edgelisturl)
        self.assertEqual(response.status_code, 200)

    def test_download_han(self):
         
        @dataclasses.dataclass
        class opts:
            root = "test_data"
            dataset = "han"
            val_seed = 2
            canonical_test_seed = 1
            val_fraction = 0.2
            test_fraction = 0.2
            

        dataset = McCallaDataset(root="test_data", hash="abc", opts=opts(), features=False)

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_download_shalek(self):

        @dataclasses.dataclass
        class opts:
            root = "test_data"
            dataset = "shalek"
            val_seed = 2
            canonical_test_seed = 1
            val_fraction = 0.2
            test_fraction = 0.2

        dataset = McCallaDataset(root="test_data", hash="abc", opts=opts(), features=False)

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_download_jackson(self):
        @dataclasses.dataclass
        class opts:
            root = "test_data"
            dataset = "jackson"
            val_seed = 2
            canonical_test_seed = 1
            val_fraction = 0.2
            test_fraction = 0.2

        dataset = McCallaDataset(root="test_data", hash="abc", opts=opts(), features=False)

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_download_zhao(self):
        @dataclasses.dataclass
        class opts:
            root = "test_data"
            dataset = "zhao"
            val_seed = 2
            canonical_test_seed = 1
            val_fraction = 0.2
            test_fraction = 0.2

        dataset = McCallaDataset(root="test_data", hash="abc", opts=opts(), features=False)

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_ping_features(self):
        # we cant download the features just for testing, so we only ping if the server is up
        response = requests.head(McCallaDataset.featuretableurl)
        self.assertEqual(response.status_code, 200)


class McCallaDatasetProcessTest(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree("src/tests/data/processed", ignore_errors=True)

    def test_preprocess_edgelist(self):
        @dataclasses.dataclass
        class opts:
            root = "test_data"
            dataset = "mccallatest"
            val_seed = 2
            canonical_test_seed = 1
            val_fraction = 0.2
            test_fraction = 0.2

        dataset = McCallaDataset(root="src/tests/data/", hash="abc", opts=opts(), features=False)
        edgelist = dataset.read_edgelist()
        self.assertEqual(edgelist.shape, (2, 5))

    def test_preprocess_features(self):

        @dataclasses.dataclass
        class opts:
            root = "test_data"
            dataset = "mccallatest"
            val_seed = 2
            canonical_test_seed = 1
            val_fraction = 0.2
            test_fraction = 0.2

        dataset = McCallaDataset(root="src/tests/data/", hash="abc", opts=opts(), features=True)
        features = dataset.read_features()

        # check if has right shape
        self.assertEqual(features.shape, (8, 3))

        # check if the node "ABCD" is sorted to the top of the list
        self.assertTrue(torch.equal(features[0, :].squeeze(), torch.FloatTensor((0.123, 0.123, 0.123))))

        edgelist = dataset.read_edgelist()

        # make sure edgelist indices and feauture sorting match, nodes 0 and 1 should have no edges
        self.assertEqual(edgelist.min(), 2)

        
    def test_full_preprocessing(self):
        @dataclasses.dataclass
        class opts:
            root = "test_data"
            dataset = "mccallatest"
            val_seed = 2
            canonical_test_seed = 1
            val_fraction = 0.2
            test_fraction = 0.2

        dataset = McCallaDataset(root="src/tests/data/", hash="abc", opts=opts(), features=True)

        # test that all pieces are there
        for key in ["train_data", "val_data", "test_data", "pot_net"]:
            self.assertTrue(hasattr(dataset, key))

        # test that in preprocessing, the graph has been converted to undirected

        self.assertEqual(dataset.train_data.edge_index.shape[1], 6)
        self.assertEqual(dataset.val_data.edge_index.shape[1], 6)
        self.assertEqual(dataset.test_data.edge_index.shape[1], 8)

        self.assertTrue(isinstance(dataset.train_data.x, torch.FloatTensor))


class GranPyDatasetTest(unittest.TestCase):

    def test_construct_pot_net(self):
        edge_index = torch.LongTensor([[0, 1, 2],
                                       [3, 4, 4]])
        
        pot_net = GranPyDataset.construct_pot_net(edge_index)

        self.assertTrue(torch.equal(pot_net[0], torch.LongTensor([[0, 1, 2],
                                                                  [4, 3, 3]])))
        self.assertTrue(torch.equal(pot_net[1], torch.LongTensor([0, 1, 2])))
        
    def test_construct_pot_net_with_ambivalent(self):    
        edge_index = torch.LongTensor([[0, 0, 1, 2],
                                       [3, 1, 4, 4]])
        
        pot_net = GranPyDataset.construct_pot_net(edge_index)

        self.assertTrue(torch.equal(pot_net[0], torch.LongTensor([[0, 1, 2, 2],
                                                                  [4, 3, 1, 3]])))
        self.assertTrue(torch.equal(pot_net[1], torch.LongTensor([0, 1, 2, 2])))

    def test_split_pot_net(self):
        neg_edges = torch.LongTensor([[0, 1, 2, 2],
                                      [4, 3, 1, 3]])
        
        masks = GranPyDataset.split_pot_net(neg_edges=neg_edges, test_fraction=0.25, val_fraction=0.25)


        self.assertEqual(masks["train"].sum() + masks["val"].sum() + masks["test"].sum(), neg_edges.shape[1])

        self.assertTrue(torch.equal(masks["train"] + masks["val"] + masks["test"], torch.ones((neg_edges.shape[1],))))

    def test_split_pot_net_reproducible(self):
        neg_edges = torch.LongTensor([[0, 1, 2, 2],
                                      [4, 3, 1, 3]])
        
        masks = GranPyDataset.split_pot_net(neg_edges=neg_edges, test_fraction=0.25, val_fraction=0.25, test_seed=1)
        masks_equal = GranPyDataset.split_pot_net(neg_edges=neg_edges, test_fraction=0.25, val_fraction=0.25, test_seed=1)
        masks_unequal = GranPyDataset.split_pot_net(neg_edges=neg_edges, test_fraction=0.25, val_fraction=0.25, test_seed=2)


        self.assertTrue(torch.equal(masks["train"], masks_equal["train"]))
        self.assertFalse(torch.equal(masks["train"], masks_unequal["train"]))


    def test_split_data(self):
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])

        data = Data(edge_index=edge_index, num_nodes=10)

        train_data, val_data, test_data = GranPyDataset.split_data(data, val_seed=1, test_seed=2, test_fraction=0.2, val_fraction=0.2)

        self.assertEqual(train_data.edge_index.shape[1], 6)  # contains only train edges
        self.assertEqual(val_data.edge_index.shape[1], 6)  # contains only train edges
        self.assertEqual(test_data.edge_index.shape[1], 8)  # contains train and val edges

    def test_split_data_same_test_different_val(self):
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])

        data = Data(edge_index=edge_index, num_nodes=10)

        train_data_1, val_data_1, test_data_1 = GranPyDataset.split_data(data, val_seed=1, test_seed=3, test_fraction=0.2, val_fraction=0.2)
        train_data_2, val_data_2, test_data_2 = GranPyDataset.split_data(data, val_seed=2, test_seed=3, test_fraction=0.2, val_fraction=0.2)

        test_edges_1 = test_data_1.edge_label_index[:, test_data_1.edge_label == 1]
        test_edges_2 = test_data_2.edge_label_index[:, test_data_2.edge_label == 1]

        self.assertTrue(torch.equal(test_edges_1, test_edges_2))

        val_edges_1 = val_data_1.edge_label_index[:, val_data_1.edge_label == 1]
        val_edges_2 = val_data_2.edge_label_index[:, val_data_2.edge_label == 1]

        self.assertFalse(torch.equal(val_edges_1, val_edges_2))


class DatasetBootstrapperTest(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree("src/tests/data/processed", ignore_errors=True)

    def test_raise_valueerror_for_nonsense_name(self):
        @dataclasses.dataclass
        class opts:
            root = "src/tests/data/"
            dataset = "nosuchdataset"

        self.assertRaises(ValueError, DatasetBootstrapper, opts=opts(), hash="abc")

    def test_finds_name(self):
        @dataclasses.dataclass
        class opts:
            root = "src/tests/data/"
            dataset = "mccallatest"

        bootstrapper = DatasetBootstrapper(opts, hash="abs")

        self.assertEqual(bootstrapper.datasetclass, McCallaDataset)

    def test_can_initialize(self):
        @dataclasses.dataclass
        class opts:
            root = "src/tests/data/"
            dataset = "mccallatest"

        dataset = DatasetBootstrapper(opts, hash="abs").get_dataset()

        self.assertTrue(isinstance(dataset, McCallaDataset))


if __name__ == '__main__':
    unittest.main(warnings='ignore')

