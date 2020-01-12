import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
from math import exp


# A deep nn that helps make decisions
class DQN(nn.Module):
    def __init__(self, width, height, num_players, outputs):
        super(DQN, self).__init__()

        self.input_num = width * height * (num_players + 1)

        self.fc1 = nn.Linear(in_features=self.input_num, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=48)
        self.out = nn.Linear(in_features=48, out_features=outputs)

    def forward(self, tensor):
        tensor = tensor.view(tensor.numel() // self.input_num, -1)

        tensor = F.relu(self.fc1(tensor))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.out(tensor)
        return tensor


# Data structure in which the memories are stored
Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


# Object storing multiple memories
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.index = 0
        self.memories = []

    def push(self, experience):
        if len(self.memories) < self.capacity:
            self.memories.append(None)

        self.memories[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def sample(self, num_samples):
        return random.sample(self.memories, num_samples)

    def __len__(self):
        return len(self.memories)


# Object representing a strategy for selecting actions
class EpsilonGreedy(object):
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def curr_threshold(self, curr_step):
        return self.end + (self.start - self.end) * exp(-1 * curr_step / self.decay)
