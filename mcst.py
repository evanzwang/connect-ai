import numpy as np
import torch
from torch import nn

from env import BoardManager


class MCST:
    def __init__(self, pvnn: nn.Module, bm: BoardManager, c_puct: float, noise_alpha: float, noise_epsilon: float,
                 device: torch.device, win_reward: float = 1, **kwargs):
        self.pvnn = pvnn
        # Store as (state, action) or state + np array
        self.q_s_a = {}
        self.n_s_a = {}
        self.p_s_a = {}
        self.n_s = {}
        self.bm = bm
        self.c = c_puct
        self.alpha = noise_alpha
        self.eps = noise_epsilon
        self.device = device
        self.win_reward = win_reward

    def reset(self):
        self.q_s_a = {}
        self.n_s_a = {}
        self.p_s_a = {}
        self.n_s = {}

    def action_probs(self, state: np.ndarray, player: int, temperature: float) -> np.ndarray:
        action_counts = self.n_s_a[self.bm.standard_perspective(state, player).tobytes()]
        if temperature == 0:
            valid_actions = action_counts * self.bm.valid_moves(state)
            action = np.random.choice(np.flatnonzero(valid_actions == valid_actions.max()))
            prob = np.zeros(action_counts.size)
            prob[action] = 1
            return prob
        temp_scaled = np.power(action_counts, 1 / temperature)
        return temp_scaled / temp_scaled.sum()

    def search(self, state: np.ndarray, player: int):
        # Returns v, whoever got it
        # Two-player "should" still work as expected

        q_val_arr = []
        n_val_arr = []
        player_arr = []
        action_arr = []

        encoded_state = self.bm.standard_perspective(state, player).tobytes()
        valid_moves = self.bm.valid_moves(state)

        reward = None

        while encoded_state in self.p_s_a:
            prob = self.p_s_a[encoded_state]
            q_val_arr.append(self.q_s_a[encoded_state])
            n_val_arr.append(self.n_s_a[encoded_state])
            player_arr.append(player)

            noise = np.random.dirichlet([self.alpha] * prob.size) * self.eps
            curr_prob = (prob + noise) * valid_moves

            puct = q_val_arr[-1] + self.c * curr_prob * np.sqrt(self.n_s[encoded_state]) / (n_val_arr[-1] + 1)
            self.n_s[encoded_state] += 1

            chosen_action = puct.argmax()
            action_arr.append(chosen_action)
            state, win_status = self.bm.take_action(state, chosen_action, player)

            if win_status:
                reward = 0 if win_status == -2 else self.win_reward
                player = win_status
                break

            valid_moves = self.bm.valid_moves(state)
            player = self.bm.next_player(player)
            encoded_state = self.bm.standard_perspective(state, player).tobytes()

        if reward is None:
            with torch.no_grad():
                nn_input = torch.from_numpy(self.bm.onehot_perspective(state, player))
                nn_input = nn_input.to(device=self.device).unsqueeze(0).float()
                prob, state_val = self.pvnn(nn_input)
            prob = prob[0].cpu().numpy()
            reward = state_val[0].cpu().numpy()
            prob *= valid_moves
            prob_sum = prob.sum()

            if prob_sum == 0:
                print("No valid moves. (hehe)")
                prob = valid_moves / valid_moves.sum()
            else:
                prob /= prob_sum

            self.p_s_a[encoded_state] = prob
            self.n_s[encoded_state] = 0
            self.n_s_a[encoded_state] = np.zeros(prob.size, dtype=np.int32)
            self.q_s_a[encoded_state] = np.zeros(prob.size)

        for i in range(len(player_arr)-1, -1, -1):
            apply_r = (-1) ** (player_arr[i] != player) * reward
            curr_action = action_arr[i]
            q_val_arr[i][curr_action] = (q_val_arr[i][curr_action] * n_val_arr[i][curr_action] + apply_r) / \
                                        (n_val_arr[i][curr_action] + 1)
            n_val_arr[i][curr_action] += 1
