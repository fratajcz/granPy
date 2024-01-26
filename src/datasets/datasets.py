from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils as pyg_utils
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, robust_scale
from sklearn.model_selection import train_test_split
import torch
import numpy as np

class GranPyDataset(InMemoryDataset):
    """ Abstract class that governs the joint behaviours of all datasets in granPy, such as datset naming an storage behaviour.
        Also governs the joint preprocessing details such as addition of inverse edges etc."""
    def __init__(self, root, hash):
        self.root = root
        self.hash = hash
        self.canonical_test_seed = 1
        self.test_fraction = 0.2
        self.val_fraction = 0.1
        super().__init__(root)

        self.data = torch.load(self.processed_paths[0])

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

        features = robust_scale(features, axis=0)

        if pyg_utils.contains_self_loops(edgelist):
            edgelist = pyg_utils.remove_self_loops(edgelist)[0]

        edgelist = pyg_utils.to_undirected(edgelist)

        # calculating the masks this way introduces trivial examples. Problem?

        train_mask, val_mask, test_mask = self.get_masks(edgelist.shape[1])

        data = Data(x=features,
                    edge_index=edgelist,
                    train_mask=torch.BoolTensor(train_mask),
                    val_mask=torch.BoolTensor(val_mask),
                    test_mask=torch.BoolTensor(test_mask))
        
        torch.save(data, self.processed_paths[0])



    def get_masks(self, num_edges):
        
        indices = np.array(range(num_edges))
        
        trainval_indices, test_indices = train_test_split(indices, test_size=self.test_fraction, random_state=self.canonical_test_seed)

        real_val_frac = self.val_fraction / (1 - self.test_fraction)

        train_indices, val_indices = train_test_split(trainval_indices, test_size=real_val_frac)

        train_mask = np.array([index in train_indices for index in indices])
        val_mask = np.array([index in val_indices for index in indices])
        test_mask = np.array([index in test_indices for index in indices])

        return train_mask, val_mask, test_mask



class McCallaDataset(GranPyDataset):
    """ Governs the download and preprocessing of the four datasets from McCalla et al. """
    def __init__(self, root, hash, name, features=True):
        self.features = features
        self.name = name

        self.name2edgelist = {"zhao": "gold_standards/mESC/mESC_chipunion.txt",
                              "jackson": "gold_standards/yeast/yeast_KDUnion.txt",
                              "shalek": "gold_standards/mDC/mDC_chipunion.txt",
                              "han": "gold_standards/hESC/hESC_chipunion.txt",
                              "test": "edgelist.csv"}
        
        self.name2featuretable = {"zhao": "expression_data/normalized/zhao_GSE114952.csv.gz",
                                  "jackson": "expression_data/normalized/jackson_GSE125162.csv.gz",
                                  "shalek": "expression_data/normalized/shalek_GSE48968.csv.gz",
                                  "han": "expression_data/normalized/han_GSE107552.csv.gz",
                                  "test": "node_features.csv.gz"}
        
        if name not in self.name2edgelist.keys():
            raise ValueError("Only datasets for zhao, jackson, shalek and han et al implemented.")
        
        super().__init__(root, hash)

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