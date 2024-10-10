import unittest
from src.utils import get_model_hash, get_dataset_hash
import dataclasses


class GetHashTest(unittest.TestCase):
    def test_hashes_are_different(self):

        @dataclasses.dataclass(unsafe_hash=True)
        class opts1:
            val_seed: int = 1
            canonical_test_seed: str = "hello"

        @dataclasses.dataclass(unsafe_hash=True)
        class opts2:
            val_seed: int = 2
            canonical_test_seed: str = "hello"

        hash1 = get_model_hash(opts1())
        hash2 = get_model_hash(opts2())

        self.assertFalse(hash1 == hash2)

    def test_hashes_are_identical(self):

        @dataclasses.dataclass
        class opts1:
            val_seed: int = 1
            canonical_test_seed: str = "hello"

        @dataclasses.dataclass
        class opts2:
            val_seed: int = 1
            canonical_test_seed: str = "hello"

        hash1 = get_model_hash(opts1())
        hash2 = get_model_hash(opts2())

        self.assertTrue(hash1 == hash2)