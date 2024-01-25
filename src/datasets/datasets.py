from torch_geometric.data import InMemoryDataset, Data
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import torch


class GranPyDataset(InMemoryDataset):
    """ Abstract class that governs the joint behaviours of all datasets in granPy, such as datset naming an storage behaviour.
        Also governs the joint preprocessing details such as addition of inverse edges etc."""
    def __init__(self, root, hash):
        self.root = root
        self.hash = hash
        super().__init__(root)

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
        data_name = self.hash + '.pt'
        train_name = self.hash + '_train.pt'
        val_name = self.hash + '_val.pt'
        return [data_name, train_name, val_name]

    def download(self):
        # Download to `self.raw_dir`.
        pass



class McCallaDataset(GranPyDataset):
    """ Governs the download and preprocessing of the four datasets from McCalla et al. """
    def __init__(self, root, hash, name, features=True):
        self.features = features
        self.name = name

        self.name2edgelist = {"zhao": "gold_standards/mESC/mESC_chipunion.txt",
                          "jackson": "gold_standards/yeast/yeast_KDUnion.txt",
                          "shalek": "gold_standards/mDC/mDC_chipunion.txt",
                          "han": "gold_standards/hESC/hESC_chipunion.txt",
                          "test": "../../src/tests/data/edgelist.csv"}
        
        self.name2featuretable = {"zhao": "expression_data/normalized/zhao_GSE114952.csv.gz",
                                  "jackson": "expression_data/normalized/jackson_GSE125162.csv.gz",
                                  "shalek": "expression_data/normalized/shalek_GSE48968.csv.gz",
                                  "han": "expression_data/normalized/han_GSE107552.csv.gz"}
        
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
            filename = os.path.join(self.raw_dir, "gold_standard_datasets.zip")
            print("Downloading features for {} from {} to {}".format(self.name, self.featuretableurl, filename))
            # TODO: Add the feature download

    @property
    def raw_file_names(self):
        files = []
        files.append(self.name2edgelist[self.name.lower()])
        if self.features:
            files.append(self.name2featuretable[self.name.lower()])
        return files

    def process(self) -> Data:
        # TODO implement dataset-specific processing like edgelist and feature reading

        print("NO CACHE FOUND - Processing data...")
    
        if self.features:
            features = self.read_features()

        edges = self.read_edgelist()

    def read_features(self) -> torch.FloatTensor:
        # TODO: add feature processing
        pass

    def read_edgelist(self) -> torch.LongTensor:
        edge_df = pd.read_csv(self.raw_paths[0], header=None, index_col=None, sep="\t")
        old_shape = edge_df.values.shape

        if hasattr(self, "geneencoder"):
            encoded_edges = self.geneencoder.transform(edge_df.values.reshape(-1, 1)).reshape(old_shape)
        else:
            self.geneencoder = OrdinalEncoder()
            encoded_edges = self.geneencoder.fit_transform(edge_df.values.reshape(-1, 1)).reshape(old_shape)

        return torch.LongTensor(encoded_edges)
    
    @classmethod
    @property
    def edgelisturl(self):
        return "https://zenodo.org/records/5909090/files/gold_standard_datasets.zip"
    
    @classmethod
    @property
    def featuretableurl(self):
        return "https://zenodo.org/records/5909090/files/expression_data.zip"