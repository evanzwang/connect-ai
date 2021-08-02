from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn


from env import BoardManager, calc_inv_equiv


class MCST:
    """
    Monte-Carlo Search Tree
    All-inclusive class holding the data of the search.
    All input states should be the actual board state, not from a certain player's perspective.
    """
    LARGE_VAL = 1000

    def __init__(self, pvnn: nn.Module, bm: BoardManager, c_puct: float, noise_alpha: float, noise_epsilon: float,
                 device: torch.device, win_reward: float = 1, **kwargs):
        """
        Initializes variables.
        :param pvnn: The NN predicting the probabilities of actions and values of the state
        :param bm: Board manager for utility functions in game rules
        :param c_puct: Hyper-parameter weighting the Q-values of actions less when selecting search paths
        :param noise_alpha: Alpha to generate Dirichlet noise
        :param noise_epsilon: Weight of the Dirichlet noise (how much the noise effects the policy)
        :param device: Device (for PyTorch), either cpu or cuda
        :param win_reward: The reward gained from winning
        """
        # Dictionary of NumPy arrays containing the Q-values of each action at a given state. (state, Q-values)
        self.q_s_a = {}
        # Dictionary of NumPy arrays containing the visited count of each action at a given state. (state, count)
        self.n_s_a = {}
        # Dictionary of NumPy arrays containing the probability of each action at a given state. (state, probability)
        self.p_s_a = {}
        # Dictionary containing the sum of the visited counts of each action at a given state (state, count)
        self.n_s = {}

        self.pvnn = pvnn
        self.bm = bm
        self.c = c_puct
        self.alpha = noise_alpha
        self.eps = noise_epsilon
        self.device = device
        self.win_reward = win_reward

    def reset(self):
        """
        Resets the MCST memory
        """
        self.q_s_a = {}
        self.n_s_a = {}
        self.p_s_a = {}
        self.n_s = {}

    def action_probs(self, state: np.ndarray, player: int, temperature: float) -> np.ndarray:
        """
        Returns the decided action probabilities (how many searches were done down that action) for a given state
        :param state: The board state, dimension [height, width]
        :param player: The player to move
        :param temperature: Hyper-parameter controlling the exploration. Higher temperature -> more "random"
        :return: The action probabilities to take
        """
        action_counts = self.n_s_a[self.bm.standard_perspective(state, player).tobytes()]
        if temperature == 0:
            action = np.random.choice(np.flatnonzero(action_counts == action_counts.max()))
            prob = np.zeros(action_counts.size)
            prob[action] = 1
            return prob
        temp_scaled = np.power(action_counts, 1 / temperature)
        return temp_scaled / temp_scaled.sum()

    def search(self, state: np.ndarray, player: int):
        """
        Runs a MC search from the current state, with the given player's turn to move
        Picks an action using the PUCT algorithm, until a new state is reached (or if the game ends).
        The NN is used to predict the value of that state and probabilities of each action.
        The value is propagated back. (If the game ends, the reward is also propagated back.)

        :param state: The board state, dimension [height, width]
        :param player: The player next to move
        """

        # Recording the NumPy arrays that need to be updated at the end when updating Q-values and counts
        q_val_arr = []
        n_val_arr = []
        player_arr = []
        action_arr = []

        encoded_state = self.bm.standard_perspective(state, player).tobytes()
        valid_moves = self.bm.valid_moves(state)

        reward = None

        while encoded_state in self.p_s_a:  # Iterating until a new state is reached
            # For updates, later
            q_val_arr.append(self.q_s_a[encoded_state])
            n_val_arr.append(self.n_s_a[encoded_state])
            player_arr.append(player)

            # Getting the probabilities of the state and adding noise
            prob = self.p_s_a[encoded_state]
            noise = np.random.dirichlet([self.alpha] * prob.size) * self.eps
            curr_prob = prob + noise
            curr_prob = curr_prob / curr_prob.sum()
            # Deciding which action to take
            puct = q_val_arr[-1] + self.c * curr_prob * np.sqrt(self.n_s[encoded_state]) / (n_val_arr[-1] + 1)
            self.n_s[encoded_state] += 1

            chosen_action = (puct - ~valid_moves*self.LARGE_VAL).argmax()
            action_arr.append(chosen_action)

            # Taking the action and updating state
            state, win_status = self.bm.take_action(state, chosen_action, player)

            if win_status:  # if the game has ended
                reward = 0 if win_status == -2 else self.win_reward
                player = win_status  # who got the reward
                break

            # Setting up the variables for the next iteration (deeper into the tree)
            valid_moves = self.bm.valid_moves(state)
            player = self.bm.next_player(player)
            encoded_state = self.bm.standard_perspective(state, player).tobytes()

        if reward is None:  # reward is None iff the state was a new one
            # Estimating the current state's action probabilities and value
            with torch.no_grad():
                nn_input, equiv = self.bm.random_equivalence(state)
                nn_input = self.bm.onehot_perspective(nn_input, player)
                nn_input = torch.from_numpy(nn_input).to(device=self.device).unsqueeze(0).float()
                prob, state_val = self.pvnn(nn_input)
            prob = prob[0].cpu().numpy().reshape(self.bm.height, self.bm.width)
            prob = calc_inv_equiv(prob, equiv).reshape(self.bm.height * self.bm.width)
            reward = state_val[0].cpu().numpy()
            # Setting values in the MCST memory for the new state
            self.p_s_a[encoded_state] = prob
            self.n_s[encoded_state] = 0
            self.n_s_a[encoded_state] = np.zeros(prob.size, dtype=np.int32)
            self.q_s_a[encoded_state] = np.zeros(prob.size)

        for i in range(len(player_arr)-1, -1, -1):  # Updating Q-values and counts
            # Is -reward if the player wasn't the player that got the win
            apply_r = (-1) ** (player_arr[i] != player) * reward
            curr_action = action_arr[i]
            q_val_arr[i][curr_action] = (q_val_arr[i][curr_action] * n_val_arr[i][curr_action] + apply_r) / \
                                        (n_val_arr[i][curr_action] + 1)
            n_val_arr[i][curr_action] += 1
