import torch
import torch.optim as optim

from classes import *


# Agent representation
class Agent(object):
    def __init__(self, strategy, num_actions=7, device="cpu", width=7, height=6, num_players=2, memory_capacity=1e5,
                 batch_size=128, gamma=0.999, learning_rate=0.001, is_learning=False):

        self.curr_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

        self.policy_net = DQN(width, height, num_players, width).to(device)

        self.is_learning = is_learning

        if is_learning:
            self.target_net = DQN(width, height, num_players, width).to(device)
            self.update_target_network()
            self.target_net.eval()

            self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
            self.gamma = gamma
            self.batch_size = batch_size
            self.memory = ReplayMemory(memory_capacity)

    def select_action(self, state):
        threshold = self.strategy.curr_threshold(self.curr_step)
        self.curr_step += 1
        if threshold < random.random():
            return torch.tensor([random.randrange(self.num_actions)]).to(self.device)
        else:
            with torch.no_grad():
                return self.policy_net(state).argmax(1).to(self.device)

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        next_states = torch.cat(batch.next_state)
        rewards = torch.cat(batch.reward)

        curr_state_values = self.policy_net(states).gather(1, actions)
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_values = (next_state_values * self.gamma) + rewards

        loss = F.mse_loss(curr_state_values, expected_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        # May not be necessary
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def push_memory(self, experience):
        self.memory.push(experience)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def load_network(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()

    def save_network(self, path):
        torch.save(self.policy_net.state_dict(), path)
