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
        for key in ["train_data", "val_data", "test_data"]:
            self.assertTrue(hasattr(dataset, key))

        self.assertEqual(dataset.train_data.edge_index.shape[1], 3)  # contains only train edges
        self.assertEqual(dataset.val_data.edge_index.shape[1], 3)  # contains only train edges
        self.assertEqual(dataset.test_data.edge_index.shape[1], 4)  # contains train and val edges
        
        self.assertEqual(dataset.train_data.known_edges.shape[1], 3)  # contains only train edges
        self.assertEqual(dataset.val_data.known_edges.shape[1], 4)  # contains train and val edges
        self.assertEqual(dataset.test_data.known_edges.shape[1], 5)  # contains all edges
        
        self.assertEqual(dataset.train_data.pos_edges.shape[1], 3)  # contains only train edges
        self.assertEqual(dataset.val_data.pos_edges.shape[1], 1)  # contains only val edges
        self.assertEqual(dataset.test_data.pos_edges.shape[1], 1)  # contains only test edges
        
        for data in ["train_data", "val_data", "test_data"]:
            self.assertTrue(getattr(getattr(dataset, data), "num_nodes"), 8)

        self.assertTrue(isinstance(dataset.train_data.x, torch.FloatTensor))


class GranPyDatasetTest(unittest.TestCase):

    def test_construct_pot_net(self):
        edge_index = torch.LongTensor([[0, 1, 2],
                                       [3, 4, 4]])
        data = Data(known_edges=edge_index,
                    num_nodes=5)

        pot_net = GranPyDataset.construct_pot_net(data)
        
        self.assertTrue(torch.equal(pot_net, torch.LongTensor([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                                                                  [1, 2, 4, 0, 2, 3, 0, 1, 3]])))
        
    def test_construct_pot_net_with_ambivalent(self):    
        edge_index = torch.LongTensor([[0, 1, 2, 4],
                                       [3, 4, 4, 2]])
        
        data = Data(known_edges=edge_index,
                    num_nodes=5)
        
        pot_net = GranPyDataset.construct_pot_net(data)

        self.assertTrue(torch.equal(pot_net, torch.LongTensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4],
                                                                  [1, 2, 4, 0, 2, 3, 0, 1, 3, 0, 1, 3]])))
        
    def test_construct_pot_net_with_isolated(self):    
        edge_index = torch.LongTensor([[0, 1, 2],
                                       [3, 4, 4]])
        
        data = Data(known_edges=edge_index,
                    num_nodes=7)
        
        pot_net = GranPyDataset.construct_pot_net(data)

        self.assertTrue(torch.equal(pot_net, torch.LongTensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                                                                  [1, 2, 4, 5, 6, 0, 2, 3, 5, 6, 0, 1, 3, 5, 6]])))

    def test_split_data(self):
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])

        data = Data(edge_index=edge_index, num_nodes=10)

        train_data, val_data, test_data = GranPyDataset.split_data(data, val_seed=1, test_seed=2, test_fraction=0.2, val_fraction=0.2)

        self.assertEqual(train_data.edge_index.shape[1], 6)  # contains only train edges
        self.assertEqual(val_data.edge_index.shape[1], 6)  # contains only train edges
        self.assertEqual(test_data.edge_index.shape[1], 8)  # contains train and val edges
        
        self.assertEqual(train_data.known_edges.shape[1], 6)  # contains only train edges
        self.assertEqual(val_data.known_edges.shape[1], 8)  # contains train and val edges
        self.assertEqual(test_data.known_edges.shape[1], 10)  # contains all edges
        
        self.assertEqual(train_data.pos_edges.shape[1], 6)  # contains only train edges
        self.assertEqual(val_data.pos_edges.shape[1], 2)  # contains only val edges
        self.assertEqual(test_data.pos_edges.shape[1], 2)  # contains only test edges
        
        for data in [train_data, val_data, test_data]:
            self.assertTrue(getattr(data, "num_nodes"), 10)

    def test_split_data_same_test_different_val(self):
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])

        data = Data(edge_index=edge_index, num_nodes=10)

        train_data_1, val_data_1, test_data_1 = GranPyDataset.split_data(data, val_seed=1, test_seed=3, test_fraction=0.2, val_fraction=0.2)
        train_data_2, val_data_2, test_data_2 = GranPyDataset.split_data(data, val_seed=2, test_seed=3, test_fraction=0.2, val_fraction=0.2)

        self.assertTrue(torch.equal(test_data_1.pos_edges, test_data_2.pos_edges))
        self.assertFalse(torch.equal(val_data_1.pos_edges, val_data_2.pos_edges))


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
            val_seed = 2
            canonical_test_seed = 1
            val_fraction = 0.2
            test_fraction = 0.2

        dataset = DatasetBootstrapper(opts, hash="abs").get_dataset()

        self.assertTrue(isinstance(dataset, McCallaDataset))


if __name__ == '__main__':
    unittest.main(warnings='ignore')

