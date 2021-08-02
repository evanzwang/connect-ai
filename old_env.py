import numpy as np


# Represents the game board and contains several methods that can be run to play the game
class Board:
    delta_list = np.array(
        [[-1, 0], [-1, 1], [0, 1], [1, 1]]
    )

    # Initializes board with set width, height, and how many tiles need to be in a row to win
    def __init__(self, width, height, connect_num):
        self.width = width
        self.height = height
        self.connect_num = connect_num
        self.winner = 0
        self.board = np.zeros((height, width), dtype=np.int8)

    # Called on a player win and returns the winner (player number)
    # I don't think this is used
    def on_win(self):
        winner = self.winner
        self.reset_board()
        return winner

    # I don't think this is used
    def on_lose(self):
        self.reset_board()
        return self.winner

    # Resets the board to all zeros
    def reset_board(self):
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.winner = 0

    # Sets the board to a pre-existing board
    def set_board(self, new_board):
        self.board = new_board
        self.height = new_board.shape[0]
        self.width = new_board.shape[1]

    # Makes a move with 5-in-a-row rules (without gravity), given an r and c index.
    # Returns -2 if all filled, -1 if the move cannot be made, 0 if the move was made, player # if the player won
    def direct_place(self, r, c, player):
        if not (0 in self.board):
            return -2

        if r >= self.height or r < 0 or c >= self.width or c < 0:
            return -1

        if self.board[r, c] == 0:
            self.board[r, c] = player
            return self.is_connected(r, c)

        return -1

    # Makes a move with Connect-4 move rules, given a position and player number.
    # Returns -2 if all filled, -1 if the move cannot be made, 0 if the move was made, player # if the player won
    def make_move(self, col, player):
        if not (0 in self.board):
            return -2

        if col < 0 or col >= self.width:
            return -1

        # current_row = self.board[pos]
        current_col = self.board[:, col]
        if current_col[0] != 0:
            return -1

        replace_r = current_col.argmin()
        current_col[replace_r] = player
        return self.is_connected(replace_r, col)
        # for i in range(self.height):
        #     if current_row[i] == 0:
        #         current_row[i] = player
        #         return self.is_connected(pos, i)

    # Checks if any player has won
    # Returns 0 if no one won, player # if player won
    def is_connected(self, r, c):
        player = self.board[r, c]
        # delta_r = 0
        # delta_c = 0

        # for x_change in range(-1, 2):
        #     for y_change in range(0, 2):
        #         delta_r = x_change
        #         delta_c = y_change
        #         if (delta_r == 1 and delta_c == 0) or (delta_r == 0 and delta_c == 0):
        #             break
        for check in self.delta_list:
            delta_r = check[0]
            delta_c = check[1]
            new_r = r + delta_r
            new_c = c + delta_c
            current_streak = 1
            # 0 is good, 1 is going to be reversed, 2 is already reversed, 3 is going to end
            reverse_state = 0

            while 1:
                if new_r < 0 or new_r >= self.height or new_c < 0 or new_c >= self.width:
                    reverse_state += 1
                    # print("hi")
                elif self.board[new_r, new_c] == player:
                    current_streak += 1
                    # print(new_r, new_c, "wefoi", delta_r, delta_c)
                    # print(reverse_state)
                else:
                    reverse_state += 1

                if reverse_state > 0:
                    if reverse_state == 1:
                        reverse_state += 1
                        delta_r = -delta_r
                        delta_c = -delta_c
                        new_r = r
                        new_c = c
                    elif reverse_state == 3:
                        break

                if current_streak >= self.connect_num:
                    self.winner = player
                    return player

                new_r += delta_r
                new_c += delta_c
        return 0


class ManualGame:
    def __init__(self, width, height, connect_num, num_players, is_direct):
        self.game_board = Board(width, height, connect_num)
        self.is_direct = is_direct
        self.num_players = num_players

    def get_move(self, current_player):
        if self.is_direct:
            can_exit = False
            while not can_exit:
                move_pos = input("Player " + str(current_player) + ": Input move position <r c>: ").strip().split()
                if len(move_pos) != 2:
                    print("Please input again, there was a problem in your input.")
                    continue

                move_status = self.game_board.direct_place(int(move_pos[0]), int(move_pos[1]), current_player)
                if move_status == 0:
                    can_exit = True
                elif move_status > 0:
                    print("Player " + str(current_player) + " won!")
                    return current_player
                elif move_status == -1:
                    print("Move cannot be made.")
                elif move_status == -2:
                    print("Board is full. Resetting.")
                    return -2
            return 0

        else:
            can_exit = False
            while not can_exit:
                move_pos = input("Player " + str(current_player) + ": Input move position: ")

                move_status = self.game_board.make_move(int(move_pos), current_player)
                if move_status == 0:
                    can_exit = True
                elif move_status > 0:
                    print("Player " + str(current_player) + " won!")
                    return current_player
                elif move_status == -1:
                    print("Move cannot be made.")
                elif move_status == -2:
                    print("Board is full. Quitting.")
                    return -2
            return 0

    def play_round(self):
        player_iter = range(1, self.num_players + 1)
        self.display()

        turn_count = 1
        while 1:
            for player in player_iter:
                move_status = self.get_move(player)
                if move_status == 0:
                    print("\nTurn " + str(turn_count) + ".")
                    self.display()
                elif move_status > 0:
                    print("Finished on turn " + str(turn_count) + ".")
                    self.display()
                    return player
                elif move_status == -2:
                    print("Filled board on turn " + str(turn_count) + ".")
                    self.display()
                    return -2

                turn_count += 1

    def display(self):
        print(self.game_board.board)


def main():
    w = h = 10
    game = ManualGame(w, h, 5, 2, True)
    game.play_round()
    # de = game.game_board.board

    # de = np.array([[1, 1, 1, 1, 2, 1, 1, 0, 0, 0]
    #                   , [2, 2, 1, 0, 0, 0, 0, 0, 0, 0]
    #                   , [2, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    #                   , [2, 0, 0, 2, 1, 0, 0, 0, 0, 0]
    #                   , [2, 0, 0, 0, 2, 2, 0, 0, 0, 0]
    #                   , [2, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    #                   , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #                   , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #                   , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #                   , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int)

    # onehot_board = np.zeros((2 + 1, w, h), dtype=de.dtype)
    # np.put_along_axis(onehot_board, np.expand_dims(de, 0), 1, axis=0)
    #
    # roll_amount = 1 - 2
    # onehot_board[1:] = np.roll(onehot_board[1:], roll_amount, axis=0)
    # print("TWO PERSP", onehot_board)
    #
    # onehot_board = np.zeros((2 + 1, w, h), dtype=de.dtype)
    # np.put_along_axis(onehot_board, np.expand_dims(de, 0), 1, axis=0)
    #
    # roll_amount = 1 - 1
    # onehot_board[1:] = np.roll(onehot_board[1:], roll_amount, axis=0)
    # print("ONE PERSP", onehot_board)

    pass


if __name__ == "__main__":
    main()

# width = 10 #7
# height = 7 #6
# connect_num = 4
# game = ManualGame(width, height, connect_num, 2, False)

# game.play_round()

# b.make_move(2, 1)
# b.make_move(3, 1)
# b.make_move(4, 1)
# b.direct_place(4, 3, 2)
# b.direct_place(3, 2, 2)
# b.direct_place(2, 1, 3)
# b.direct_place(1, 0, 2)
# b.direct_place(6, 5, 2)

# b.direct_place(5, 4, 2)
# b.make_move(1, 1)
# b.make_move(1, 1)
# b.make_move(0, 1)