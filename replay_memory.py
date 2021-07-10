import numpy as np

from collections import deque
import random


class ReplayMemory:
    def __init__(self):
        self.memories = deque()
        raise NotImplementedError

    def store_memory(self, state, next_state, action, reward, done):
        raise NotImplementedError

    def sample(self, size: int):
        return random.sample(self.memories, size)
