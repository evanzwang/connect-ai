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
            action = np.random.choice(np.flatnonzero(action_counts == action_counts.max()))
            prob = np.zeros(action_counts.size)
            prob[action] = 1
            return prob
        temp_scaled = np.power(action_counts, 1 / temperature)
        return temp_scaled / temp_scaled.sum()

    def search(self, state: np.ndarray, player: int) -> tuple[float, int]:
        # Returns v, whoever got it
        # Two-player "should" still work as expected

        s_encode = self.bm.standard_perspective(state, player).tobytes()
        valid_moves = self.bm.valid_moves(state)
        if s_encode not in self.p_s_a:
            with torch.no_grad():
                nn_input = torch.from_numpy(self.bm.onehot_perspective(state, player))
                nn_input = nn_input.to(device=self.device).unsqueeze(0).float()
                prob, state_val = self.pvnn(nn_input)
            prob = prob[0].cpu().numpy()
            state_val = state_val[0].cpu().numpy()
            prob *= valid_moves
            prob_sum = prob.sum()

            if prob_sum == 0:
                print("No valid moves. (hehe)")
                prob = valid_moves / valid_moves.sum()
            else:
                prob /= prob_sum

            self.p_s_a[s_encode] = prob
            self.n_s[s_encode] = 0
            self.n_s_a[s_encode] = np.zeros(prob.size, dtype=np.int32)
            self.q_s_a[s_encode] = np.zeros(prob.size)

            return state_val, player

        prob = self.p_s_a[s_encode]
        q_vals = self.q_s_a[s_encode]
        n_vals = self.n_s_a[s_encode]

        noise = np.random.dirichlet([self.alpha] * prob.size) * self.eps
        curr_prob = (prob + noise) * valid_moves

        puct = q_vals + self.c * curr_prob * np.sqrt(self.n_s[s_encode]) / (n_vals + 1)
        self.n_s[s_encode] += 1

        chosen_action = puct.argmax()
        new_state, win_status = self.bm.take_action(state, chosen_action, player)

        if win_status:
            reward = 0 if win_status == -2 else self.win_reward
            apply_v = (-1) ** (player != win_status) * reward
            q_vals[chosen_action] = (q_vals[chosen_action] * n_vals[chosen_action] + apply_v) / \
                                    (n_vals[chosen_action] + 1)
            n_vals[chosen_action] += 1
            return reward, win_status

        next_player = self.bm.next_player(player)

        next_v, v_player = self.search(new_state, next_player)
        apply_v = (-1) ** (player != v_player) * next_v
        # Recomputing average
        q_vals[chosen_action] = (q_vals[chosen_action] * n_vals[chosen_action] + apply_v) / (n_vals[chosen_action] + 1)
        n_vals[chosen_action] += 1

        return next_v, v_player