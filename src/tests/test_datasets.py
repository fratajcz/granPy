import unittest
from src.datasets.datasets import McCallaDataset
import pathlib
import shutil
import os
import requests
import torch

class McCallaDatasetDownloadTest(unittest.TestCase):
    def setUp(self):
        pathlib.Path("test_data").mkdir(parents=True, exist_ok=True)    

    def tearDown(self):
        shutil.rmtree("test_data", ignore_errors=True)

    def test_ping_edgelist(self):
        response = requests.head(McCallaDataset.edgelisturl)
        self.assertEqual(response.status_code, 200)

    def test_download_han(self):
        dataset = McCallaDataset(root="test_data", hash="abc", name="han", features=False)

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_download_shalek(self):
        dataset = McCallaDataset(root="test_data", hash="abc", name="shalek", features=False)

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_download_jackson(self):
        dataset = McCallaDataset(root="test_data", hash="abc", name="jackson", features=False)

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_download_zhao(self):
        dataset = McCallaDataset(root="test_data", hash="abc", name="zhao", features=False)

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_ping_features(self):
        # we cant download the features just for testing, so we only ping if the server is up
        response = requests.head(McCallaDataset.featuretableurl)
        self.assertEqual(response.status_code, 200)

class McCallaDatasetProcessTest(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree("src/tests/data/processed", ignore_errors=True)

    def test_preprocess_edgelist(self):
        dataset = McCallaDataset(root="src/tests/data/", hash="abc", name="test", features=False)
        edgelist = dataset.read_edgelist()
        print(edgelist)
        self.assertEqual(edgelist.shape, (2, 3))

    def test_preprocess_features(self):
        dataset = McCallaDataset(root="src/tests/data/", hash="abc", name="test", features=True)
        features = dataset.read_features()

        # check if has right shape
        self.assertEqual(features.shape, (8, 3))

        # check if the node "ABCD" is sorted to the top of the list
        self.assertTrue(torch.equal(features[0, :].squeeze(), torch.FloatTensor((0.123, 0.123, 0.123))))

        edgelist = dataset.read_edgelist()

        # make sure edgelist indices and feauture sorting match, nodes 0 and 1 should have no edges
        self.assertEqual(edgelist.min(), 2)

        
    def test_full_preprocessing(self):
        dataset = McCallaDataset(root="src/tests/data/", hash="abc", name="test", features=True)


        # test that all pieces are there
        for key in ["x", "edge_index", "test_mask", "train_mask", "val_mask"]:
            self.assertTrue(key in dataset.data.keys)

        # test that all edges are assigned to one of the sets
        self.assertEqual(dataset.edge_index.shape[1], dataset.test_mask.sum() + dataset.train_mask.sum() + dataset.val_mask.sum())

        # test that in preprocessing, the graph has been converted to undirected

        self.assertEqual(dataset.edge_index.shape[1], 6)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
