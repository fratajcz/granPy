import unittest
from src.utils import get_hash
import dataclasses


class GetHashTest(unittest.TestCase):
    def test_hashes_are_different(self):

        @dataclasses.dataclass
        class opts1:
            some_key = 1
            other_key = "hello"

        @dataclasses.dataclass
        class opts2:
            some_key = 2
            other_key = "hello"

        hash1 = get_hash(opts1())
        hash2 = get_hash(opts2())

        self.assertFalse(hash1 == hash2)