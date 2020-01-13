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

EPS_START = 0.96
EPS_END = 0.008
EPS_DECAY = 608
MEMORY_CAPACITY = 16384
LEARNING_RATE = 0.001
GAMMA = 0.999
BATCH_SIZE = 256

NUM_EPISODES = 2048
TARGET_UPDATE = 12
SWITCH_AGENT = 24
USE_RANDOM_THRESH = 0.32
REWARD_ARR = [0.0, 10000.0, -100.0, -0.1]
FILE_PATH = "nn_models/episode_"

WIN_RATE_HISTORY = 100
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
    if len(record_t) >= WIN_RATE_HISTORY:
        means = record_t.unfold(0, WIN_RATE_HISTORY, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(WIN_RATE_HISTORY-1), means))
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
    for episode in range(1, NUM_EPISODES + 1):
        env.reset()
        current_player = random.randrange(0, NUM_PLAYERS) % NUM_PLAYERS
        env.set_player(current_player)

        for _ in count():
            state = torch.from_numpy(env.render_perspective(current_player)).to(device).float()

            action = agents[current_player].select_action(state)
            # print(agents[current_player].strategy.curr_threshold(agents[current_player].curr_step))

            _, reward, done, _ = env.step(action.item())

            if current_player == 0:
                next_state = torch.from_numpy(env.render_perspective(0)).to(device).float()
                reward_t = torch.tensor([reward], device=device)
                training_agent.push_memory(Experience(state, action, next_state, reward_t))

                training_agent.optimize()

            if done:
                print(training_agent.strategy.curr_threshold(training_agent.curr_step))
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

            switching_time = episode / (TARGET_UPDATE * SWITCH_AGENT)
            if int(switching_time) == switching_time:

                full_path = FILE_PATH + str(episode) + ".pth"
                training_agent.save_network(full_path)

                old_strat = EpsilonGreedy(USE_RANDOM_THRESH, 0, 1)
                for i in range(1, len(agents)):
                    agents[i] = Agent(old_strat, **hyperparameters)
                    agents[i].load_network(full_path)

                training_agent.reset_strategy()
                training_agent.strategy = EpsilonGreedy(
                    (EPS_START * (1 - USE_RANDOM_THRESH) ** (1.6 * switching_time)) + EPS_END, EPS_END, EPS_DECAY
                )

    print("Done")
    plt.ioff()
    plt.show()


main()
