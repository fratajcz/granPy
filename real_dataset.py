from src.datasets.datasets import McCallaDataset

dataset = McCallaDataset(root="/mnt/storage/granpy/data/", hash="abc", name="han")

print(dataset.train_data)