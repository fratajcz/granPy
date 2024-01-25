from torch_geometric.data import InMemoryDataset
import os

class GranPyDataset(InMemoryDataset):
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
        return ['{}.pt'.format(self.hash)]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # TODO implement dataset-independent processing like adding inverse edges etc


class McCallaDataset(GranPyDataset):
    def __init__(self, root, hash, name):
        self.name = name
        self.name2path = {"zhao": "gold_standards/mESC/mESC_chipunion.txt",
                          "jackson": "gold_standards/yeast/yeast_KDUnion.txt",
                          "shalek": "gold_standards/mDC/mDC_chipunion.txt",
                          "han": "gold_standards/hESC/hESC_chipunion.txt"}
        super().__init__(root, hash)

    def download(self):
        from urllib.request import urlretrieve
        import zipfile

        url = "https://zenodo.org/records/5909090/files/gold_standard_datasets.zip"
        filename = os.path.join(self.raw_dir, "gold_standard_datasets.zip")
        print("Downloading dataset {} from {} to {}".format(self.name, url, filename))
        urlretrieve(url, filename)

        with zipfile.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

    @property
    def raw_file_names(self):
        return [self.name2path[self.name.lower()]]
    
    def process(self):
        # TODO implement dataset-specific processing like edgelist reading
    