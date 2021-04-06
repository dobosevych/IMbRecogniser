import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from skimage import io, transform
import numpy as np


class IMbDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.imb_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.imb_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        сolumns = self.imb_frame.columns.to_list()
        img_name = os.path.join(self.root_dir,
                                self.imb_frame.iloc[idx, сolumns.index('image')])
        image = io.imread(img_name)
        s = self.imb_frame.iloc[idx, сolumns.index('sequence')]
        label = np.array([{"A": 1, "D": 2, "F": 3, "T": 4}[c] for c in s])

        sample = image

        if self.transform:
            sample = self.transform(image)

        return sample, label