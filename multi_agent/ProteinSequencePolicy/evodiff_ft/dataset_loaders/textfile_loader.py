"""Basic loader for a set of sequences in a text file, intended
as input to the evodiff models for fine-tuning."""
from torch.utils.data import Dataset
import numpy as np
import pandas as pd



class TextfileDataset(Dataset):
    """Imports sequences from a text file which may have multiple columns but
    where the sequences are in the first column only, and the remaining columns
    can be disregarded. Also the first row is disregarded."""

    def __init__(self, fpath, max_len=np.inf):
        self.data = pd.read_csv(fpath).iloc[:,0].tolist()
        self.indices = list(range(len(self.data)))
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        if len(sequence) > self.max_len:
            start = np.random.choice(len(sequence) - self.max_len)
            stop = start + self.max_len
            sequence = sequence[start:stop]
        return (sequence,)
