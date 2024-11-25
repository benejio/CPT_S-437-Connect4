#Connect4.py

import numpy as np

# Constants
ROWS = 6
COLUMNS = 7
PLAYER_ONE = 1
PLAYER_TWO = -1
EMPTY = 0

class Connect4:
    def __init__(self):
        self.board = np.zeros((ROWS, COLUMNS), dtype=int)
        self.current_player = PLAYER_ONE

    def reset(self):
        """Resets the game board for a new game."""
        self.board = np.zeros((ROWS, COLUMNS), dtype=int)
        self.current_player = PLAYER_ONE

    def make_move(self, col):
        """Places a piece in the specified column for the current player, filling from the bottom."""
        if not self.is_valid_location(col):
            print("Invalid move. Column is full.")
            return False
    
        # Start from the bottom row and move upwards
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = self.current_player
                return True
        return False

    def is_valid_location(self, col):
        """Checks if the column has at least one empty cell."""
        return self.board[0][col] == EMPTY

    def switch_player(self):
        """Switches the current player."""
        self.current_player *= -1

    def check_winner(self):
        """Checks if the current player has won."""
        for row in range(ROWS):
            for col in range(COLUMNS - 3):
                if np.all(self.board[row, col:col + 4] == self.current_player):
                    return True
        for col in range(COLUMNS):
            for row in range(ROWS - 3):
                if np.all(self.board[row:row + 4, col] == self.current_player):
                    return True
        for row in range(ROWS - 3):
            for col in range(COLUMNS - 3):
                if all(self.board[row + i][col + i] == self.current_player for i in range(4)):
                    return True
                if all(self.board[row + 3 - i][col + i] == self.current_player for i in range(4)):
                    return True
        return False

    def print_board(self):
        """Prints the board in a readable format."""
        print(self.board)

# # Example Game Simulation
# game = Connect4()
# game_over = False

# while not game_over:
#     game.print_board()
#     try:
#         col = int(input(f"Player {game.current_player}, select a column (0-{COLUMNS-1}): "))
#         if col < 0 or col >= COLUMNS:
#             print(f"Invalid input. Please enter a number between 0 and {COLUMNS - 1}.")
#             continue
#     except ValueError:
#         print("Invalid input. Please enter a valid integer.")
#         continue
    
#     if game.make_move(col):
#         if game.check_winner():
#             game.print_board()
#             print(f"Player {game.current_player} wins!")
#             game_over = True
#         else:
#             game.switch_player()
#     else:
#         print("Invalid move. Try again.")
