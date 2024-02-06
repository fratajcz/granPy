import unittest
from src.utils import get_hash
import dataclasses


class GetHashTest(unittest.TestCase):
    def test_hashes_are_different(self):

        @dataclasses.dataclass(unsafe_hash=True)
        class opts1:
            some_key: int = 1
            other_key: str = "hello"

        @dataclasses.dataclass(unsafe_hash=True)
        class opts2:
            some_key: int = 2
            other_key: str = "hello"

        hash1 = get_hash(opts1())
        hash2 = get_hash(opts2())

        self.assertFalse(hash1 == hash2)

    def test_hashes_are_identical(self):

        @dataclasses.dataclass
        class opts1:
            some_key: int = 1
            other_key: str = "hello"

        @dataclasses.dataclass
        class opts2:
            some_key: int = 1
            other_key: str = "hello"

        hash1 = get_hash(opts1())
        hash2 = get_hash(opts2())

        self.assertTrue(hash1 == hash2)