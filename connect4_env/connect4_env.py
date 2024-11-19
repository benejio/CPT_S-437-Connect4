import gym
from gym import spaces
import numpy as np

class Connect4Env(gym.Env):

    '''
    Notes:
    In the connect 4 board, 0's represent empty spaces, 1's and -1's represent player tokens.
    In a standard game of connect4 the board is 6 rows by 7 column.
    Action space or decision space is what the ai can do. In this case the ai can place a token in 1 of 7 columns
    Observational space is the game board. Here I use openai gym spaces
    '''
    def __init__(self):
        super(Connect4Env, self).__init__()
        self.board_rows = 6
        self.board_cols = 7
        self.action_space = spaces.Discrete(self.board_cols)  # 7 columns to drop token in

        # Flatten the observation space to a 1D vector
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.board_rows * self.board_cols,), dtype=int
        )
        self.reset()


    '''
    Required function used for openAI gym.
    Resets the connect4 environment to its initial state at the start of a new game.

    input:
        seed (int, optional): Seed for random number generator
        options (dict, optional): Additional options for reset customization. I did not use them.

    return:
        - gameboard: The initial game board as a flattened array.
        - info (dict): Additional information about the environment. (required by openAI gym)
    
    '''
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_rows, self.board_cols), dtype=int)
        self.current_player = 1
        self.done = False
        return self.board.flatten(), {}


    '''
    Executes a player's action, updates the board state, and checks for win conditions.

    Parameters:
        action: The column (0-indexed) where the current player chooses to place their piece.

    Returns:
        gameboard: The updated game board state as a flattened 1D array.
        reward: Reward based on the outcome of the move(we might have to tweak this in order to get good results):
            Positive (e.g., 1) if the player wins.
            Zero (0) if the game continues or results in a draw.
            Negative (e.g., -10) if the move is illegal.
        done (bool): True if the game has ended (win, draw, or illegal move), False otherwise.
            Might want to change this so the game doesnt end on an illegal move?
        truncated (bool): Required output by openai gym. I think it is a required return but we do not use it.
        info (dict): Additional information. We do not use.
    '''
    def step(self, action):
        # Check if the action column is already filled
        if self.board[0, action] != 0:
            return self.board.flatten(), -10, True, False, {}  # Penalize illegal move

        # Drop the piece in the specified column
        row = self._get_available_row(action)
        self.board[row, action] = self.current_player

        # Check if the current player won
        if self._check_win(row, action):
            reward = 1  # Reward for winning
            self.done = True
        elif np.all(self.board != 0):
            reward = 0  # Draw
            self.done = True
        else:
            reward = 0  # No reward for ongoing games
            self.current_player *= -1  # Switch turns
            self.done = False

        return self.board.flatten(), reward, self.done, False, {}


    '''
    Given a column returns the lowest open row that a token will fall into
    
    input:
        col - the column that is checked

    return:
        int - the row index of the lowest open space for that column
        ValueError - if a column is full a value error will be raised
    '''
    def _get_available_row(self, col):
        for row in range(self.board_rows - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        raise ValueError("Column is full")


    '''
    Checks the board for a win after a player(ai or human) places a piece at row, col

    input:
        self - self
        row - the row where the most recent token was placed
        col - the col where the most recent token was placed
    
    return:
        bool - True if current player has won, false otherwise
    '''
    def _check_win(self, row, col):
        # Horizontal, vertical, and diagonal checks
        return (
            self._check_line(row, col, 1, 0) or  # Horizontal
            self._check_line(row, col, 0, 1) or  # Vertical
            self._check_line(row, col, 1, 1) or  # Diagonal /
            self._check_line(row, col, 1, -1)    # Diagonal \
        )

    '''
    Given the location of a board space and delta values checks for a 4-in-a-row

    input:
        row - the row where the most recent token was placed
        col - the col where the most recent token was placed
        delta_row - the row direction of movement the move on each step
            ex. 0 = no horizontal movement and 1 = rightward horizonal movement
        delta_col - the col direction of movment to move on each step
            ex. 0 = no vertical movement and 1 = upward horizontal movement
    return:
        bool - A True if a 4-in-a-row is found, False otherwise
        After checking a line going forward the algorithm goes back 
        checking each token to make sure no 4-in-a-rows were missed
    '''
    def _check_line(self, row, col, delta_row, delta_col):
        # Check a line for 4 consecutive same-player pieces
        piece = self.board[row, col]
        count = 1

        # Check in one direction
        r, c = row + delta_row, col + delta_col
        while 0 <= r < self.board_rows and 0 <= c < self.board_cols and self.board[r, c] == piece:
            count += 1
            r += delta_row
            c += delta_col

        # Check in the opposite direction
        r, c = row - delta_row, col - delta_col
        while 0 <= r < self.board_rows and 0 <= c < self.board_cols and self.board[r, c] == piece:
            count += 1
            r -= delta_row
            c -= delta_col

        return count >= 4

    '''
    Prints the current board to the consol.

    input:
        mode - Sepcifies the render mode, defaults to human. Currently not used but may be useful later.
    
    print output:
        board is 6 rows by 7 columns and each space is represented by and int
        - 0's represent empty spaces
        - 1's represent spaces with player 1's token
        - -1's represnt spaces with player 2's token
    '''
    def render(self, mode="human"):
        print(self.board)
