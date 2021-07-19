import numpy as np
import torch
from torch import nn

import abc
import random

from mcst import MCST
from env import BoardManager
# peen

def play_baseline(model: nn.Module, baseline_m: nn.Module, device: torch.device, train_dict: dict, vs_dict: dict,
                  num_trials: int = 10):
    model.eval()
    bm = BoardManager(**train_dict)
    nn_player = NNPlayer(1, bm, model, device, **train_dict)
    base_player = NNPlayer(2, bm, baseline_m, device, **vs_dict)
    g = Game([nn_player, base_player], bm)

    tot_wins = 0

    for _ in range(num_trials):
        g.reset_players()
        g.scramble_players()
        result = g.run_game()
        if result == 1:
            tot_wins += 2
        elif result == 0:
            tot_wins += 1

    # Account for treating wins = 2, draw = 1
    return tot_wins / (2 * num_trials)


def play_random(model: nn.Module, device: torch.device, num_trials: int = 10, **kwargs):
    model.eval()
    bm = BoardManager(**kwargs)
    nn_player = NNPlayer(1, bm, model, device, **kwargs)
    rand_player = RandomPlayer(2, bm)
    g = Game([nn_player, rand_player], bm)

    tot_wins = 0

    for _ in range(num_trials):
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
    model.eval()
    bm = BoardManager(**kwargs)
    nn_player = NNPlayer(1, bm, model, device, **kwargs)
    rand_player = HumanPlayer(2, bm)
    g = Game([nn_player, rand_player], bm)

    g.reset_players()
    g.scramble_players()
    result = g.run_game()
    print(f"Winner: {result}")


class Player(abc.ABC):
    @abc.abstractmethod
    def __init__(self, player: int, bm: BoardManager):
        self.player = player
        self.bm = bm

    @abc.abstractmethod
    def make_action(self, state: np.ndarray) -> int:
        pass

    def reset(self):
        pass


class Game:
    def __init__(self, players: list[Player], b_manager: BoardManager):
        self.players = players
        self.bm = b_manager

    def scramble_players(self):
        random.shuffle(self.players)

    def reset_players(self):
        for p in self.players:
            p.reset()

    # Returns who won: player # or 0 if draw
    def run_game(self):
        board = self.bm.blank_board()

        while 1:
            for p in self.players:
                action = p.make_action(board)
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
        self.tree.reset()

    def make_action(self, state: np.ndarray) -> int:
        for _ in range(self.steps):
            self.tree.search(state, self.player)

        a_prob = self.tree.action_probs(state, self.player, 0)
        action = np.random.choice(a_prob.size, p=a_prob)
        return action


class RandomPlayer(Player):
    def __init__(self, player: int, bm: BoardManager):
        super(RandomPlayer, self).__init__(player, bm)

    def make_action(self, state: np.ndarray) -> int:
        valid_moves = self.bm.valid_moves(self.bm.standard_perspective(state, self.player))
        return np.random.choice(valid_moves.size, p=valid_moves/valid_moves.sum())


class HumanPlayer(Player):
    def __init__(self, player: int, bm: BoardManager):
        super(HumanPlayer, self).__init__(player, bm)

    def make_action(self, state: np.ndarray) -> int:
        print(state)
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
