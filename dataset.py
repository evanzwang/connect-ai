import numpy as np
from torch.utils.data import Dataset
import torch

import random


class MemoryDataset(Dataset):
    def __init__(self, max_memory: int, random_replacement: bool, **kwargs):
        self.data = []
        self.curr_ind = 0
        self.max_size = max_memory
        self.random_override = random_replacement

    def __len__(self) -> int:
        return max(len(self.data), 1)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, float]:
        if self.__len__() == 1 and len(self.data) == 0:
            raise NotImplementedError
        return self.data[idx]

    def add(self, item: tuple[np.ndarray, np.ndarray, float]):
        if len(self.data) < self.max_size:
            self.data.append(item)
        else:
            if not self.random_override:
                self.data[self.curr_ind] = item
                self.curr_ind = (self.curr_ind + 1) % self.max_size
            else:
                self.data[random.randrange(0, self.max_size)] = item
