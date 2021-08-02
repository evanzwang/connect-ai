import numpy as np
import torch
from torch import nn

import abc
import random

from env import BoardManager
from mcst import MCST


def play_baseline(model: nn.Module, baseline_m: nn.Module, device: torch.device, train_dict: dict, vs_dict: dict,
                  num_trials: int = 10) -> float:
    """
    Pits the current model vs a previous baseline model. Records the win rate of the current model over num_trials games
    :param model: The current NN model
    :param baseline_m: The baseline model
    :param device: Device (for PyTorch), either cpu or cuda
    :param train_dict: Config parameters for the current NN model
    :param vs_dict: Config parameters for the baseline NN model
    :param num_trials: The number of trials to measure win rate
    :return: Average win percentage. 0 if the current lost, 1 if the current won. Draws are counted as 0.5
    """
    model.eval()
    bm = BoardManager(**train_dict)
    nn_player = NNPlayer(1, bm, model, device, **train_dict)
    base_player = NNPlayer(2, bm, baseline_m, device, **vs_dict)
    g = Game([nn_player, base_player], bm)

    tot_wins = 0

    for _ in range(num_trials):
        # Resets and scrambles order for fair play
        g.reset_players()
        g.scramble_players()
        result = g.run_game()
        if result == 1:
            tot_wins += 2
        elif result == 0:
            tot_wins += 1

    # Account for treating wins = 2, draw = 1
    return tot_wins / (2 * num_trials)


def play_random(model: nn.Module, device: torch.device, num_trials: int = 10, **kwargs) -> float:
    """
    Pits a NN model against a random player, recording the win ratio.
    :param model: The NN model
    :param device: Device (for PyTorch), either cpu or cuda
    :param num_trials: The number of trials to be played
    :return: Average win percentage. 0 NN lost, 1 if NN won. Draws are counted as 0.5
    """
    model.eval()
    bm = BoardManager(**kwargs)
    nn_player = NNPlayer(1, bm, model, device, **kwargs)
    rand_player = RandomPlayer(2, bm)
    g = Game([nn_player, rand_player], bm)

    tot_wins = 0

    for _ in range(num_trials):
        # Resets and scrambles order for fair play
        g.reset_players()
        g.scramble_players()
        result = g.run_game()
        if result == 1:
            tot_wins += 2
        elif result == 0:
            tot_wins += 1

    # Account for treating wins = 2, draw = 1
    return tot_wins / (2 * num_trials)


def play_model_human(model: nn.Module, device: torch.device, **kwargs):
    """
    Pits an NN model versus a manual/human player. One game via console
    :param model: The NN model
    :param device: Device (for PyTorch), either cpu or cuda
    """
    model.eval()
    bm = BoardManager(**kwargs)
    nn_player = NNPlayer(1, bm, model, device, **kwargs)
    rand_player = HumanPlayer(2, bm)
    g = Game([nn_player, rand_player], bm)
    # Resets and scrambles the order of players
    g.reset_players()
    g.scramble_players()
    result = g.run_game()
    print(f"Winner: {result}")


class Player(abc.ABC):
    """
    Abstract class for NNPlayer, RandomPlayer, and HumanPlayer
    """
    @abc.abstractmethod
    def __init__(self, player: int, bm: BoardManager):
        self.player = player
        self.bm = bm

    @abc.abstractmethod
    def choose_action(self, state: np.ndarray) -> int:
        """
        Chooses an action based on the given board state. (This is the main difference between players)
        :param state: The board state, dimension [height, width]
        :return: The action
        """
        pass

    def reset(self):
        pass


class Game:
    """
    Game class that assembles a list of players and runs games using those players
    """
    def __init__(self, players: list[Player], b_manager: BoardManager):
        """
        :param players: List of Players
        :param b_manager: The board manager with game specifications
        """
        self.players = players
        self.bm = b_manager

    def scramble_players(self):
        """
        Scrambles the list of Players (used before a game to randomize playing order)
        """
        random.shuffle(self.players)

    def reset_players(self):
        """
        Resets each player (only used for NN players)
        """
        for p in self.players:
            p.reset()

    def run_game(self) -> int:
        """
        Runs a game between the list of players, until a player wins or there is a draw.
        :return: The winner. If drawn, returns 0
        """
        board = self.bm.blank_board()

        while 1:
            # Iterates through all the players in order
            for p in self.players:
                action = p.choose_action(board)
                board, win_status = self.bm.take_action(board, action, p.player)
                if win_status != 0:
                    return win_status if win_status > 0 else 0


class NNPlayer(Player):
    def __init__(self, player: int, bm: BoardManager, model: nn.Module, device: torch.device, mcst_steps: int,
                 c_puct: float = 1, **kwargs):
        super(NNPlayer, self).__init__(player, bm)
        self.model = model
        self.model.eval()
        self.steps = mcst_steps
        self.tree = MCST(model, bm, c_puct, 0.003, 0, device)
        self.bm = bm

    def reset(self):
        # Resets the MCST
        self.tree.reset()

    def choose_action(self, state: np.ndarray) -> int:
        # Runs a Monte Carlo Tree Search a given number of times
        for _ in range(self.steps):
            self.tree.search(state, self.player)

        # Selects the action with maximum probability
        a_prob = self.tree.action_probs(state, self.player, 0)
        action = np.random.choice(a_prob.size, p=a_prob)
        return action


class RandomPlayer(Player):
    def __init__(self, player: int, bm: BoardManager):
        super(RandomPlayer, self).__init__(player, bm)

    def choose_action(self, state: np.ndarray) -> int:
        valid_moves = self.bm.valid_moves(self.bm.standard_perspective(state, self.player))
        return np.random.choice(valid_moves.size, p=valid_moves/valid_moves.sum())


class HumanPlayer(Player):
    def __init__(self, player: int, bm: BoardManager):
        super(HumanPlayer, self).__init__(player, bm)

    def choose_action(self, state: np.ndarray) -> int:
        print(state)
        # Input validation is not perfect, so be careful in inputs.
        if self.bm.is_direct:
            while 1:
                move_pos = input("Player " + str(self.player) + ": Input move position <r c>: ").strip().split()
                if len(move_pos) != 2:
                    print("Please input again, there was a problem in your input.")
                    continue
                if state[int(move_pos[0]), int(move_pos[1])] == 0:
                    return int(move_pos[0]) * self.bm.width + int(move_pos[1])
                else:
                    print("Move is invalid.")
        else:
            while 1:
                move_pos = input("Player " + str(self.player) + ": Input move position: ").strip()
                if state[0, int(move_pos)] == 0:
                    return int(move_pos)
                else:
                    print("Move is invalid.")
