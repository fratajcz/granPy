import unittest
from src.datasets.datasets import McCallaDataset
import pathlib
import shutil
import os
import requests

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
    def setUp(self):
        pathlib.Path("test_data").mkdir(parents=True, exist_ok=True)    

    def tearDown(self):
        shutil.rmtree("test_data", ignore_errors=True)

    def test_preprocess_edgelist(self):
        dataset = McCallaDataset(root="test_data", hash="abc", name="test", features=False)

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

if __name__ == '__main__':
    unittest.main(warnings='ignore')
