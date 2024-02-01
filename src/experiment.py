from src.datasets import DatasetBootstrapper
from src.utils import get_hash


class Experiment:
    def __init__(self, opts):
        self.opts = opts

        self.hash = get_hash(opts)

        self.dataset = DatasetBootstrapper(opts, hash=self.hash).get_dataset()