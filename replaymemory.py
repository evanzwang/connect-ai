from collections import namedtuple
import random

Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
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