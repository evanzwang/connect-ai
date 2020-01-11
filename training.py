import torch

from itertools import count

from env import ConnectEnv
from classes import *
from agent import Agent


# Hyper-parameters
NUM_PLAYERS = 2
NUM_ACTIONS = 7
WIDTH = 7
HEIGHT = 6
CONNECT_NUM = 4

EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 0.01
MEMORY_CAPACITY = 100000
LEARNING_RATE = 0.001
GAMMA = 0.999

BATCH_SIZE = 128
NUM_EPISODES = 300
TARGET_UPDATE = 10
REWARD_ARR = [0.1, 1000, -0.1, -0.1]

# Initializes agent and memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    "num_actions": NUM_ACTIONS, "device": device, "width": WIDTH, "height": HEIGHT,
    "num_players": NUM_PLAYERS, "memory_capacity": MEMORY_CAPACITY, "batch_size": BATCH_SIZE, "gamma": GAMMA,
    "learning_rate": LEARNING_RATE,
}

training_agent = Agent(
    EpsilonGreedy(EPS_START, EPS_END, EPS_DECAY), hyperparameters, is_learning=True
)

random_agent = Agent(EpsilonGreedy(1, 1, 1), hyperparameters)

agents = [training_agent, random_agent]

env = ConnectEnv(WIDTH, HEIGHT, CONNECT_NUM, NUM_PLAYERS, REWARD_ARR)


# Training processes
for episode in range(NUM_EPISODES):
    env.reset()
    current_player = env.current_player
    state = env.render_perspective(current_player)
    for t in count():
        action = training_agent.select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        next_state = env.render_perspective(current_player)

        state = next_state
        training_agent.push_memory(Experience(state, action, next_state, reward))

        if done:
            break

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        current_player = env.current_player + 1

print("Done")

# Testing
print(torch.randn(4))
print("test")
print("test2")
