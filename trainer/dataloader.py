import glob
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_directory, freq=2000):
        self.data_directory = data_directory
        self.files = []
        for directory in self.data_directory:
            self.files.extend(glob.glob(directory + "*.pt"))
        self.size = len([name for name in self.files])
        self.loaded = {}

    def __len__(self):
        return self.size

    def __getitem__(self, idx, scale=False):
        items = []
        single = False
        if not isinstance(idx, list):
            idx = [idx]
            single = True
        for idx_single in idx:
            if idx_single not in self.loaded.keys():
                content = torch.load(self.files[idx_single])
                self.loaded[idx_single] = content
            items.append(self.loaded[idx_single].tolist())
        if single:
            return torch.tensor(items[0]).float()
        return torch.tensor(items).float()

