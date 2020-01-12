import torch
import matplotlib
import matplotlib.pyplot as plt

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
EPS_DECAY = 512
MEMORY_CAPACITY = 5000
LEARNING_RATE = 0.001
GAMMA = 0.999

BATCH_SIZE = 128
NUM_EPISODES = 500
TARGET_UPDATE = 10
REWARD_ARR = [0.0, 10000.0, -0.1, -0.1]

# Initializes agent and memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    "num_actions": NUM_ACTIONS, "device": device, "width": WIDTH, "height": HEIGHT,
    "num_players": NUM_PLAYERS, "memory_capacity": MEMORY_CAPACITY, "batch_size": BATCH_SIZE, "gamma": GAMMA,
    "learning_rate": LEARNING_RATE,
}

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def plot_wins(record):
    plt.figure(2)
    plt.clf()
    record_t = torch.tensor(record, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Wins/Draws/Loses')
    plt.plot(record_t.numpy())
    # Take 100 episode averages and plot them too
    if len(record_t) >= 100:
        means = record_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def main():
    training_agent = Agent(
        EpsilonGreedy(EPS_START, EPS_END, EPS_DECAY), **hyperparameters, is_learning=True
    )

    random_agent = Agent(EpsilonGreedy(1, 1, 1), **hyperparameters)

    agents = [training_agent, random_agent]

    env = ConnectEnv(WIDTH, HEIGHT, CONNECT_NUM, NUM_PLAYERS, REWARD_ARR)
    record = []

    # Training processes
    for episode in range(NUM_EPISODES):
        env.reset()
        current_player = env.current_player
        for _ in count():
            state = torch.from_numpy(env.render_perspective(current_player)).to(device).float()

            action = agents[current_player].select_action(state)

            _, reward, done, _ = env.step(action.item())

            if current_player == 0:
                next_state = torch.from_numpy(env.render_perspective(0)).to(device).float()
                reward_t = torch.tensor([reward], device=device)
                training_agent.push_memory(Experience(state, action, next_state, reward_t))

                training_agent.optimize()

            if done:
                if reward == REWARD_ARR[1]:
                    won = 0.5
                else:
                    won = 0
                if current_player != 0:
                    won *= -1
                won += 0.5

                record.append(won)
                plot_wins(record)
                break

            current_player = env.current_player

        if episode % TARGET_UPDATE == 0:
            training_agent.update_target_network()

    print("Done")
    plt.ioff()
    plt.show()


main()
