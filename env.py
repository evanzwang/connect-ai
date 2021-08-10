import numpy as np

import random


def calc_equivalence(state: np.ndarray, equiv_num: int) -> np.ndarray:
    """
    Returns an equivalent board state given a specified rotation/reflection by equiv_num
    :param state: The board state, dimensions of [height, width]
    :param equiv_num: 0 - original, 1 - flipped horizontal, 2 - orig rot 90, 3 - flipped rot 90, 4 - orig rot 180 deg
    (valid for 0 -> 7, inclusive)
    :return: An equivalent board, as specified by equiv_num
    """

    # If odd, means that the board is reflected
    if equiv_num % 2:
        state = np.fliplr(state)
    return np.rot90(state, equiv_num // 2)


def calc_inv_equiv(state: np.ndarray, equiv_num: int) -> np.ndarray:
    """
    Calculates the inverse board equivalence (rotates back and then flips, based on equiv_num)
    :param state: The board state, dimensions of [height, width]
    :param equiv_num: 0 - original, 1 - flipped horizontal, 2 - orig rot 90, 3 - flipped rot 90, 4 - orig rot 180 deg
    (valid for 0 -> 7, inclusive)
    :return: Inverse function of calc_equivalence. Will revert an equivalent board to the original board
    """
    state = np.rot90(state, -(equiv_num // 2))
    if equiv_num % 2:
        state = np.fliplr(state)
    return state


class BoardManager:
    """
    Utility functions for specified board rules
    """
    delta_list = np.array(
        [[-1, 0], [-1, 1], [0, 1], [1, 1]]
    )

    def __init__(self, height: int, width: int, connect_num: int, is_direct: bool, num_players: int, **kwargs):
        self.height = height
        self.width = width
        self.connect_num = connect_num
        self.is_direct = is_direct
        self.num_players = num_players
        # Valid equivalences
        if not self.is_direct:
            self.valid_equivs = [0, 1]  # Only horizontal flip
        elif self.height != self.width:
            self.valid_equivs = [0, 1, 4, 5]  # Combinations of horizontal flip and 180 deg rotation
        else:
            self.valid_equivs = [0, 1, 2, 3, 4, 5, 6, 7]

    def random_equivalence(self, state: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Gets a random equivalence of the given state
        :param state: The board state, dimensions of [height, width]
        :return: A random equivalence with dimension [height, width], as well as the chosen applied equiv_num
        """
        chosen = np.random.choice(self.valid_equivs)
        return calc_equivalence(state, chosen), chosen

    def all_equivalences(self, state: np.ndarray, probabilities: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Returns a shuffled list of all valid equivalences, given board and probability arrays
        :param state: The board state, dimensions of [height, width]
        :param probabilities: The probabilities for each action, dimensions of [num_actions]
        :return: The list of all equivalences applied to the board state and probabilities
        """
        if not self.is_direct:
            raise NotImplementedError
        # Shuffles the list for random ordering
        random.shuffle(self.valid_equivs)
        equivs = []
        for i in self.valid_equivs:
            equiv_state = calc_equivalence(state, i)
            # Reshapes 1D array into 2D for rotation and reflection
            equiv_prob = calc_equivalence(probabilities.reshape(self.height, self.width), i)
            equivs.append((equiv_state, equiv_prob.reshape(self.height * self.width)))
        return equivs

    def blank_board(self) -> np.ndarray:
        """
        Returns a blank board (for resetting the game)
        :return: A NumPy uint8 array with all 0's (no moves yet), dimension [height, width]
        """
        return np.zeros((self.height, self.width), dtype=np.uint8)

    def next_player(self, player: int) -> int:
        """
        Returns the next player to move (for 3 player: 1 -> 2, 2 -> 3, 3 -> 1)
        :param player: The current player
        :return: The next player to move
        """
        return player % self.num_players + 1

    # Returns updated state and whether the game terminated (who won) | assumes action is valid
    def take_action(self, state: np.ndarray, action: int, player: int) -> tuple[np.ndarray, int]:
        """
        Returns a new board with the specified action and player applied to it, as well as whether a player has won
        If is_direct, then the action means the player places a piece in the action//width row and action % width column
        If not, the action simply directs which column the player drops the piece
        :param state: The board state, dimension [height, width]
        :param action: The specified action
        :param player: The player who is taking the action
        :return: The altered board, and an int representing the win status. 0 - no result, 1 - player 1 won,
        2 - player 2 won, etc.
        -2 is used if there is a draw.
        """
        new_state = state.copy()
        if self.is_direct:
            new_state[action // self.width, action % self.width] = player
            winning_player = self.is_connected(new_state, action // self.width, action % self.width)
        else:
            replace_r = new_state[:, action].argmin()
            new_state[replace_r, action] = player
            winning_player = self.is_connected(new_state, replace_r, action)

        return new_state, winning_player

    def valid_moves(self, state: np.ndarray) -> np.ndarray:
        """
        Returns all valid moves of the given board
        :param state: The board state, dimension [height, width]
        :return: Boolean NumPy array with dimension [height, width], representing whether a player can make a move there
        """
        if self.is_direct:
            return (state == 0).reshape(self.height * self.width)
        else:
            return state[0] == 0

    def is_connected(self, state: np.ndarray, r: int, c: int) -> int:
        """
        Checks if a player has won or if there is a draw
        Assumes that a win can only result because of the newly played piece
        :param state: The board state, dimension [height, width]
        :param r: The row position of the newest piece
        :param c: The column position of the newest piece
        :return: 0 if no result, -2 if draw, 1 if P1 won, 2 if P2 won, etc.
        """
        # The player who played the piece to be checked
        player = state[r, c]

        for check in self.delta_list:
            # All 4 directions, up, right, up-right, down-right
            delta_r = check[0]
            delta_c = check[1]
            new_r = r + delta_r
            new_c = c + delta_c
            current_streak = 1
            # First check in one direction. If reaches edge or empty/enemy square, reverse direction (negative deltas).
            reverse_state = 0

            while 1:
                if new_r < 0 or new_r >= self.height or new_c < 0 or new_c >= self.width:  # Reached edge
                    reverse_state += 1
                elif state[new_r, new_c] == player:  # Continues streak
                    current_streak += 1
                else:  # Empty or enemy square
                    reverse_state += 1

                if reverse_state > 0:
                    if reverse_state == 1:
                        # Reverses checking direction
                        reverse_state += 1
                        delta_r = -delta_r
                        delta_c = -delta_c
                        new_r = r
                        new_c = c
                    elif reverse_state == 3:
                        break

                if current_streak >= self.connect_num:
                    return player
                # Next position to check
                new_r += delta_r
                new_c += delta_c
        return -2 if np.count_nonzero(self.valid_moves(state)) == 0 else 0  # Draw if all spots filled

    def onehot_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
        """
        Returns the state in the given player's perspective of the board, with one-hot encoding
        Channel 0 represents if a space is empty
        If player == 2, then channel 1 represents if a space is occupied by player 2

        Equivalent to onehot_perspective(standard_perspective(state, player), 1)
        :param state: The board state, dimension [height, width]
        :param player: The player who is next to move
        :return: NumPy array dimensions [num_players+1, height, width], with the first channel representing whether a
        space is empty. Each subsequent channel i represents whether a space is occupied by the i-th player next to move
        """
        onehot_board = np.zeros((self.num_players + 1, self.height, self.width), dtype=bool)
        np.put_along_axis(onehot_board, np.expand_dims(state, 0), True, axis=0)

        # Rolls the channel dimension to correspond to whoever is next to move
        roll_amount = 1 - player
        onehot_board[1:] = np.roll(onehot_board[1:], roll_amount, axis=0)
        return onehot_board

    def standard_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
        """
        Returns the state in the given player's perspective of the board
        All empty squares remain 0.
        Non-empty squares are represented by how many turns that player moves in
        If player == 2, then all squares equal to 2 will be changed to 1.
        :param state: The board state, dimension [height, width]
        :param player: The player who is next to move
        :return: NumPy array dimensions [height, width]. 0 means empty square. 1 means the player who is next to move
        occupies that square. 2 means the next-next player to move occupies that square.
        """
        new_state = state.copy()
        roll_amount = 1 - player
        # Fancy modular arithmetic (applied on all non-zero squares) to change perspectives
        new_state[new_state != 0] = (new_state[new_state != 0] + roll_amount - 1) % self.num_players + 1
        return new_state
