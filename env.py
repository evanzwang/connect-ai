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
        self.board = np.zeros((width, height), dtype=np.int8)

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
        self.board = np.zeros((self.width, self.height), dtype=np.int8)
        self.winner = 0

    # Sets the board to a pre-existing board
    def set_board(self, new_board):
        self.board = new_board
        self.width = new_board.size[0]
        self.height = new_board.size[1]

    # Makes a move with 5-in-a-row rules (without gravity), given an x and y index.
    # Returns -2 if all filled, -1 if the move cannot be made, 0 if the move was made, player # if the player won
    def direct_place(self, x, y, player):
        if not (0 in self.board):
            return -2

        if self.board[x, y] == 0:
            self.board[x, y] = player
            return self.is_connected(x, y)

        return -1

    # Makes a move with Connect-4 move rules, given a position and player number.
    # Returns -2 if all filled, -1 if the move cannot be made, 0 if the move was made, player # if the player won
    def make_move(self, pos, player):
        if not (0 in self.board):
            return -2

        current_row = self.board[pos]
        if current_row[self.height-1] != 0:
            return -1

        for i in range(self.height):
            if current_row[i] == 0:
                current_row[i] = player
                return self.is_connected(pos, i)

    # Checks if any player has won
    # Returns 0 if no one won, player # if player won
    def is_connected(self, x, y):        
        player = self.board[x, y]
        # delta_x = 0
        # delta_y = 0

        # for x_change in range(-1, 2):
        #     for y_change in range(0, 2):
        #         delta_x = x_change
        #         delta_y = y_change
        #         if (delta_x == 1 and delta_y == 0) or (delta_x == 0 and delta_y == 0):
        #             break
        for check in self.delta_list:
            delta_x = check[0]
            delta_y = check[1]
            new_x = x + delta_x
            new_y = y + delta_y
            current_streak = 1
            # 0 is good, 1 is going to be reversed, 2 is already reversed, 3 is going to end
            reverse_state = 0

            while 1:
                if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
                    reverse_state += 1
                    # print("hi")
                elif self.board[new_x, new_y] == player:
                    current_streak += 1
                    # print(new_x, new_y, "wefoi", delta_x, delta_y)
                    # print(reverse_state)
                else:
                    reverse_state +=1

                if reverse_state > 0:
                    if reverse_state == 1:
                        reverse_state += 1
                        delta_x = -delta_x
                        delta_y = -delta_y
                        new_x = x
                        new_y = y
                    elif reverse_state == 3:
                        break
                    
                if current_streak >= self.connect_num:
                    self.winner = player
                    return player
                    
                new_x += delta_x
                new_y += delta_y
        return 0


# Represents the environment with simple methods to run it
# Currently played in Connect-4 rules
class ConnectEnv():
    # Initializes environment with the width, height, number of tiles needed to be in-a-row, and the number of players
    def __init__(self, width, height, connect_num, num_players):
        self.state = Board(width, height, connect_num)
        self.num_players = num_players
        self.current_player = 0
        self.width = width
        self.height = height

    # Takes the action given by the player and updates the environment
    # Returns the state, reward, done, and info
    def step(self, action):
        move_status = self.state.make_move(action, self.current_player + 1)
        self.current_player = (self.current_player + 1) % self.num_players

        reward = 0
        done = False
        info = ""

        if move_status == 0:
            pass
        elif move_status > 0:            
            reward = 10
            done = True
            self.reset()
        elif move_status == -1:
            reward = -1000
        elif move_status == -2:
            reward = 0
            done = True
            self.reset()
        
        return self.state.board, reward, done, info

    # Resets the board
    def reset(self):
        self.state.reset_board()
        self.current_player = 0

    # Returns the given player's perspective of the board with one-hot encoding
    # Say if board[0,0] was 2 and given player was 2. Then the returned array[0,0] would be [0, 0, 1]
    # with the 1 representing that player 2 has taken that spot.
    def render_perspective(self, player):
        onehot_board = np.zeros((self.width, self.height, self.num_players + 1), dtype=self.state.board.dtype)

        for i in range(self.width):
            for j in range(self.height):
                occupied_state = self.state.board[i, j]
                if occupied_state == player:
                    occupied_state = 1
                elif occupied_state == 1:
                    occupied_state = player
                onehot_board[i, j, occupied_state] = 1

        return onehot_board


class ManualGame():
    def __init__(self, width, height, connect_num, num_players, is_direct):
        self.game_board = Board(width, height, connect_num)
        self.is_direct = is_direct
        self.num_players = num_players
    
    def get_move(self, current_player):
        if self.is_direct:
            can_exit = False
            while not can_exit:
                move_pos = input("Player " + str(current_player) + ": Input move position <x y>: ").strip().split()
                if len(move_pos) < 2:
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
        if self.is_direct:
            print(self.game_board.board)
        else:
            print(np.rot90(self.game_board.board))

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
