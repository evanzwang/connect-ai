import numpy as np
import torch
from torch import nn

import abc
import random

from mcst import MCST
from env import BoardManager


class Player(abc.ABC):
    @abc.abstractmethod
    def __init__(self, player: int, bm: BoardManager):
        self.player = player
        self.bm = bm

    @abc.abstractmethod
    def make_action(self, state: np.ndarray) -> int:
        pass


class Game:
    def __init__(self, players: list[Player], width: int, height: int, connect_num: int, is_direct: bool):
        self.players = players
        self.bm = BoardManager(width, height, connect_num, is_direct, len(players))

    def scramble_players(self):
        random.shuffle(self.players)

    # Returns who won: player # or 0 if draw
    def run_game(self):
        board = self.bm.blank_board()

        while 1:
            for i, p in enumerate(self.players, 1):
                action = p.make_action(board, i)
                board, win_status = self.bm.take_action(board, action, i)
                if win_status != 0:
                    return win_status if win_status > 0 else 0


class NNPlayer(Player):
    def __init__(self, player: int, bm: BoardManager, model: nn.Module, device: torch.device, mcst_steps: int,
                 c_puct: float = 1, noise_alpha: float = 0, noise_epsilon: float = 0,):
        super(NNPlayer, self).__init__(player, bm)
        self.model = model
        self.model.eval()
        self.steps = mcst_steps
        self.tree = MCST(model, bm, c_puct, noise_alpha, noise_epsilon, device)
        self.bm = bm

    def reset(self):
        self.tree.reset()

    def make_action(self, state: np.ndarray) -> int:
        curr_state = self.bm.standard_perspective(state, self.player)
        for _ in range(self.steps):
            self.tree.search(curr_state, self.player)

        a_prob = self.tree.action_probs(curr_state, self.player, 0)
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
                if state[int(move_pos[0]), int(move_pos[1])] != 0:
                    return int(move_pos[0]) * self.bm.width + int(move_pos[1])
        else:
            while 1:
                move_pos = input("Player " + str(self.player) + ": Input move position: ").strip()
                if state[0, int(move_pos)] == 0:
                    return int(move_pos)
