from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data import Dataset

class MTSDataset(Dataset):
    def __init__(self, mvts_paths):
        self.mvts_paths = mvts_paths
    def __getitem__(self, index):
        mvts_path = self.mvts_paths[index]
        mvts_value = np.load(mvts_path)
        return torch.tensor(mvts_value, dtype=torch.float32)
    def __len__(self):
        return len(self.mvts_paths)

def data_loader(data_file):
    with open(data_file, "r") as f:
        paths = f.read().split("\n")
        set = MTSDataset(mvts_paths=paths)
        if "train" in data_file:
            loader = DataLoader(set, batch_size=32, shuffle=True)
        else:
            loader = DataLoader(set, batch_size=1, shuffle=False)
    return loader