from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils as pyg_utils
from torch_geometric import seed_everything
from torch_geometric.transforms import RandomLinkSplit
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, minmax_scale
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import random
from typing import Tuple

class GranPyDataset(InMemoryDataset):
    """ Abstract class that governs the joint behaviours of all datasets in granPy, such as datset naming an storage behaviour.
        Also governs the joint preprocessing details such as addition of inverse edges etc."""
    def __init__(self, root, opts, hash):
        self.root = root
        self.hash = hash
        self.val_seed = opts.val_seed
        self.canonical_test_seed = opts.canonical_test_seed
        self.test_fraction = opts.test_fraction
        self.val_fraction = opts.val_fraction
        super().__init__(root)

        self.train_data, self.val_data, self.test_data, self.pot_net = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        pass
    
    @property
    def processed_file_names(self):
        return [self.hash + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        edgelist = self._data.edge_index
        features = self._data.x

        features = torch.FloatTensor(minmax_scale(features, axis=0))

        if pyg_utils.contains_self_loops(edgelist):
            edgelist = pyg_utils.remove_self_loops(edgelist)[0]

        data = Data(x=features,
                    edge_index=edgelist,
                    num_nodes=features.shape[0])

        train_data, val_data, test_data = self.split_data(data, self.test_fraction, self.val_fraction, self.canonical_test_seed, self.val_seed)

        pot_net = self.construct_pot_net(edgelist)

        torch.save([train_data, val_data, test_data, pot_net], self.processed_paths[0])

    @classmethod
    def split_data(self, data, test_fraction=0.2, val_fraction=0.2, test_seed=None, val_seed=None):

        data = data.clone()
        
        if val_seed is None:
            val_seed = random.randint(0, 100)

        if test_seed is None:
            test_seed = random.randint(0, 100)

        seed_everything(test_seed)

        traintest_split = RandomLinkSplit(num_val=0, num_test=test_fraction, is_undirected=False, add_negative_train_samples=True)

        trainval_data, _, test_data = traintest_split(data)

        real_val_frac = val_fraction / (1 - test_fraction)

        seed_everything(val_seed)

        trainval_split = RandomLinkSplit(num_val=real_val_frac, num_test=0, is_undirected=False, add_negative_train_samples=True)

        data.edge_index = trainval_data.edge_index

        train_data, val_data, _ = trainval_split(data)

        return train_data, val_data, test_data

    @classmethod
    def construct_pot_net(self, all_pos_edges: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor, ]:
        """ Constructs the potential network between transcription factors and targets by connecting each TF to each target
            careful, has memory complexity of n*m where n is the number of tfs and m is the number of targets

            Returns only negative edges and a tf mask by which the edges can be batched to their tfs.
            """

        all_pos_edges = pyg_utils.coalesce(all_pos_edges)

        tfs, num_targets_per_tf = all_pos_edges[0, :].unique(return_counts=True)

        targets = all_pos_edges[1, :].unique()

        tfs_repeated = tfs.repeat_interleave(repeats=targets.shape[0]).unsqueeze(0)

        tf_batches = tfs.repeat_interleave(repeats=targets.shape[0])

        targets_tiled = targets.tile((tfs.shape[0],)).unsqueeze(0)

        all_pot_net_edges, tf_batches = pyg_utils.remove_self_loops(torch.vstack((tfs_repeated, targets_tiled)), edge_attr=tf_batches)
        all_pot_net_edges, tf_batches = pyg_utils.coalesce(all_pot_net_edges, edge_attr=tf_batches)

        mask = torch.cat((tf_batches, -1 * torch.ones((all_pos_edges.shape[1],))))
        reduced_edges, pos_and_batch_mask = pyg_utils.coalesce(torch.hstack((all_pot_net_edges, all_pos_edges)), mask, reduce="mul")

        pos_mask = pos_and_batch_mask > 0

        neg_edges = reduced_edges[:, pos_mask]
        batch_mask = pos_and_batch_mask[pos_mask]

        return [neg_edges, batch_mask.abs(), tfs]

    def to(self, device):
        # for data_name in ["train_data", "val_data", "test_data", "pot_net"]:
        #    self.__setattr__(data_name, self.__getattr__(data_name).to(device))
        self.train_data = self.train_data.to(device)
        self.test_data = self.test_data.to(device)
        self.val_data = self.val_data.to(device)
        self.pot_net[0] = self.pot_net[0].to(device)
        self.pot_net[1] = self.pot_net[1].to(device)
        self.pot_net[2] = self.pot_net[2].to(device)


class McCallaDataset(GranPyDataset):
    """ Governs the download and preprocessing of the four datasets from McCalla et al. """
    def __init__(self, root, opts, hash, features=True):
        self.features = features
        self.name = opts.dataset

        self.name2edgelist = {"zhao": "gold_standards/mESC/mESC_chipunion.txt",
                              "jackson": "gold_standards/yeast/yeast_KDUnion.txt",
                              "shalek": "gold_standards/mDC/mDC_chipunion.txt",
                              "han": "gold_standards/hESC/hESC_chipunion.txt",
                              "mccallatest": "edgelist.csv"}
        
        self.name2featuretable = {"zhao": "expression_data/normalized/zhao_GSE114952.csv.gz",
                                  "jackson": "expression_data/normalized/jackson_GSE125162.csv.gz",
                                  "shalek": "expression_data/normalized/shalek_GSE48968.csv.gz",
                                  "han": "expression_data/normalized/han_GSE107552.csv.gz",
                                  "mccallatest": "node_features.csv.gz"}
        
        if self.name not in self.name2edgelist.keys():
            raise ValueError("Only datasets for zhao, jackson, shalek and han et al implemented.")
        
        super().__init__(root, opts, hash)

    def download(self) -> None:
        from urllib.request import urlretrieve
        import zipfile

        filename = os.path.join(self.raw_dir, "gold_standard_datasets.zip")
        print("Downloading edgelist for {} from {} to {}".format(self.name, self.edgelisturl, filename))
        urlretrieve(self.edgelisturl, filename)

        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        if self.features:
            filename = os.path.join(self.raw_dir, "expression_data.zip")
            print("Downloading features for {} from {} to {}".format(self.name, self.featuretableurl, filename))
            urlretrieve(self.featuretableurl, filename)
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)

    @property
    def raw_file_names(self):
        files = []
        files.append(self.name2edgelist[self.name.lower()])
        if self.features:
            files.append(self.name2featuretable[self.name.lower()])
        return files

    def process(self) -> None:

        print("NO CACHE FOUND - Processing data...")

        if self.features:
            features = self.read_features()

        edges = self.read_edgelist()

        if not self.features:
            features = torch.eye(edges.max() + 1)

        self._data = Data(x=features,
                          edge_index=edges)

        super().process()

    def read_features(self) -> torch.FloatTensor:
        feature_df = pd.read_csv(self.raw_paths[1], compression="gzip")
        self.geneencoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        feature_df["Cell"] = self.geneencoder.fit_transform(feature_df["Cell"].values.reshape(-1, 1)).flatten()

        feature_df.set_index("Cell", inplace=True, drop=True)
        feature_df.sort_index(inplace=True)

        return torch.FloatTensor(feature_df.values)

    def read_edgelist(self) -> torch.LongTensor:
        edge_df = pd.read_csv(self.raw_paths[0], header=None, index_col=None, sep="\t")

        old_shape = edge_df.values.shape

        if hasattr(self, "geneencoder"):
            encoded_edges = self.geneencoder.transform(edge_df.values.reshape(-1, 1)).reshape(old_shape)
            encoded_edges = encoded_edges[~np.isnan(encoded_edges).any(axis=1)]
        else:
            self.geneencoder = OrdinalEncoder()
            encoded_edges = self.geneencoder.fit_transform(edge_df.values.reshape(-1, 1)).reshape(old_shape)

        return torch.LongTensor(encoded_edges).T
    
    @classmethod
    @property
    def edgelisturl(self):
        return "https://zenodo.org/records/5909090/files/gold_standard_datasets.zip"
    
    @classmethod
    @property
    def featuretableurl(self):
        return "https://zenodo.org/records/5909090/files/expression_data.zip"
    
    @classmethod
    @property
    def names(self):
        return ["zhao", "jackson", "shalek", "han", "mccallatest"]


class DatasetBootstrapper:
    def __init__(self, opts, hash):
        import src.datasets as datasets

        self.hash = hash
        self.opts = opts
        name = opts.dataset
        names = []
        self.datasetclass = None

        for possible_class in map(datasets.__dict__.get, datasets.__all__):
            try:
                names.extend(possible_class.names)
                if name in possible_class.names:
                    if self.datasetclass is None:
                        self.datasetclass = possible_class
                    else:
                        raise ValueError("Found at least two implementations of dataset with name {}: {} and {}".format(name, possible_class, self.datasetclass))
            except AttributeError:
                continue

        if self.datasetclass is None:
            raise ValueError("Found no class that implements a dataset with the name {}. Did you mean one of the following: {}".format(name, names))

    def get_dataset(self):
        return self.datasetclass(root=self.opts.root, hash=self.hash, opts=self.opts)
