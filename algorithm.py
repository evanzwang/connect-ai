import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import random
import math
from collections import namedtuple
from itertools import count


from env import ConnectEnv


class DQN(nn.Module):
    def __init__(self, width, height, num_players, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=width*height*(num_players+1), out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=48)
        self.out = nn.Linear(in_features=48, out_features=outputs)

    def forward(self, tensor):
        tensor = tensor.flatten(start_dim=1)
        tensor = F.relu(self.fc1(tensor))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.out(tensor)
        return tensor


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


class EpsilonGreedy(object):
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def curr_threshold(self, curr_step):
        return self.end + (self.start - self.end) * math.exp(-1 * curr_step / self.decay)


class Agent(object):
    def __init__(self, strategy, num_actions, device):
        self.strategy = strategy
        self.num_actions = num_actions
        self.curr_step = 0
        self.device = device
    
    def select_action(self, state, policy_net):
        threshold = self.strategy.curr_threshold(self.curr_step)
        self.curr_step += 1
        if threshold < random.random():
            return torch.tensor([random.randrange(self.num_actions)]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(1).to(self.device)


# nice

NUM_PLAYERS = 2
WIDTH = 7
HEIGHT = 6
CONNECT_NUM = 4

EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 0.01
MEMORY_CAPACITY = 100000
LEARNING_RATE = 0.001

EPISODES = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(EpsilonGreedy(EPS_START, EPS_END, EPS_DECAY), WIDTH, device)
memory = ReplayMemory(MEMORY_CAPACITY)

policy_net = DQN(WIDTH, HEIGHT, NUM_PLAYERS, WIDTH).to(device)
target_net = DQN(WIDTH, HEIGHT, NUM_PLAYERS, WIDTH).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=LEARNING_RATE)

env = ConnectEnv(WIDTH, HEIGHT, CONNECT_NUM, NUM_PLAYERS)

for episode in range(EPISODES):
    env.reset()
    for t in count():
        pass



print(torch.randn(4))
print("test")
print("test2")
