import unittest
from src.datasets.datasets import McCallaDataset
import pathlib
import shutil
import os

class McCallaDatasetTest(unittest.TestCase):
    def setUp(self):
        pathlib.Path("test_data").mkdir(parents=True, exist_ok=True)    

    def tearDown(self):
        shutil.rmtree("test_data", ignore_errors=True)

    def test_download_han(self):
        dataset = McCallaDataset(root="test_data", hash=None, name="han")

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_download_shalek(self):
        dataset = McCallaDataset(root="test_data", hash=None, name="shalek")

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_download_jackson(self):
        dataset = McCallaDataset(root="test_data", hash=None, name="jackson")

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

    def test_download_zhao(self):
        dataset = McCallaDataset(root="test_data", hash=None, name="zhao")

        self.assertTrue(os.path.isfile(os.path.join(dataset.raw_dir, dataset.raw_file_names[0])))

if __name__ == '__main__':
    unittest.main(warnings='ignore')
