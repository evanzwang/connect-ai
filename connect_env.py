import numpy as np

from env import Board


# Represents the environment with simple methods to run it
class ConnectEnv:
    # Initializes environment with the width, height, number of tiles needed to be in-a-row, and the number of players
    def __init__(self, width: int, height: int, connect_num: int, num_players: int, rewards: list[int],
                 is_direct: bool, **kwargs):
        self.state = Board(width, height, connect_num)
        self.num_players = num_players
        self.current_player = 1
        self.width = width
        self.height = height
        self.rewards = rewards
        self.is_direct = is_direct

    # Takes the action given by the player and updates the environment
    # Returns the state, reward, done, and info
    def step(self, action: int) -> tuple[np.ndarray, int, bool, str]:
        if self.is_direct:
            move_status = self.state.direct_place(action // self.width, action % self.width, self.current_player)
        else:
            move_status = self.state.make_move(action, self.current_player)

        self.current_player = self.current_player % self.num_players + 1

        reward = 0
        done = False
        info = ""

        # Regular move
        if move_status == 0:
            reward = self.rewards[0]
        # Winning move
        elif move_status > 0:
            reward = self.rewards[1]
            done = True
            self.reset()
        # Illegal move
        elif move_status == -1:
            reward = self.rewards[2]
        # Drawing move
        elif move_status == -2:
            reward = self.rewards[3]
            done = True
            self.reset()

        return self.state.board, reward, done, info

    # Resets the board
    def reset(self) -> None:
        self.state.reset_board()
        self.current_player = 1

    def set_player(self, player: int) -> None:
        self.current_player = player

    # Returns the given player's perspective of the board with one-hot encoding
    # Array with dimensions [players, width, height], channel 0 marks a 1 if there is no piece
    # Channel 1 marks a 1 if the player's piece is there, everything else is shifted
    def render_perspective(self, player: int) -> np.ndarray:
        onehot_board = np.zeros((self.num_players + 1, self.width, self.height), dtype=self.state.board.dtype)
        np.put_along_axis(onehot_board, np.expand_dims(self.state.board, 0), 1, axis=0)

        roll_amount = 1 - player
        onehot_board[1:] = np.roll(onehot_board[1:], roll_amount, axis=0)
        return onehot_board
