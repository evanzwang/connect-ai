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

NUM_EPISODES = 512
REWARD_ARR = [0.0, 10000.0, -0.1, -0.1]
FILE_PATH = "nn_models/episode_"
FINAL_MODEL = 960

WIN_RATE_HISTORY = 150

# Initializes agent and memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    "num_actions": NUM_ACTIONS, "device": device, "width": WIDTH, "height": HEIGHT, "num_players": NUM_PLAYERS,
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
    if len(record_t) >= WIN_RATE_HISTORY:
        means = record_t.unfold(0, WIN_RATE_HISTORY, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(WIN_RATE_HISTORY-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def main():
    testing_agent = Agent(
        EpsilonGreedy(0, 0, 1), **hyperparameters
    )
    testing_agent.load_network(FILE_PATH + str(FINAL_MODEL) + ".pth")

    random_agent = Agent(EpsilonGreedy(1, 1, 1), **hyperparameters)

    agents = [testing_agent, random_agent]

    env = ConnectEnv(WIDTH, HEIGHT, CONNECT_NUM, NUM_PLAYERS, REWARD_ARR)
    record = []

    # Testing
    for episode in range(1, NUM_EPISODES + 1):
        env.reset()
        current_player = env.current_player
        for _ in count():
            state = torch.from_numpy(env.render_perspective(current_player)).to(device).float()

            action = agents[current_player].select_action(state)
            # print(agents[current_player].strategy.curr_threshold(agents[current_player].curr_step))

            _, reward, done, _ = env.step(action.item())

            if done:
                print(testing_agent.strategy.curr_threshold(testing_agent.curr_step))
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

    print("Done")
    plt.ioff()
    plt.show()


main()
