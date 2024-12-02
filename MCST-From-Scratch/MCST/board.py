import numpy as np
from node import Node

class Board:
    '''
        Deals with current game board
        Has helper functions to make moves and generate nodes
    '''

    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.full((rows, cols), None, dtype=object)
        self.column_heights = np.full(cols, rows - 1)


    def is_board_full(self):
        '''
            Returns true if the board is full
        '''
        return not np.any(self.board == None)

    
    def __find_clear_space(self, col):
        '''
            Returns the row that is next in drop order
        '''
        if self.column_heights[col] < 0:  # If column is full
            return None
        return self.column_heights[col]

   
    def drop_token(self, token, col):
        '''
            Drops a token at the specified column, 
                if the column is full or column is invalid returns False
        '''
        if col < 0 or col >= self.cols:
            return False
        row = self.__find_clear_space(col)
        if row is not None:
            self.board[row][col] = token
            self.column_heights[col] -= 1
            return True
        return False 

    def remove_token(self, col):
        '''
            Removes the topmost token from a given column
        '''
        if col < 0 or col >= self.cols:
            return False
        for i in range(self.rows):
            if self.board[i][col] is not None:
                self.board[i][col] = None
                self.column_heights[col] += 1
                return True

        return False 

    def generate_general_board_id(self):
        """
        Generates a generalized version of the board ID where:
        - Irrelevant tokens (cannot contribute to a Connect 4) are replaced with '0'.
        - Empty spaces (None) are represented as '.'.
        """
        rows, cols = self.board.shape
        
        def is_relevant_token(row, col):
            """
            Checks if the token at (row, col) is relevant for a future Connect 4.
            Includes blank spaces in the potential line count.
            """
            if self.board[row, col] is None:
                return False
            
            token = self.board[row, col]
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, and two diagonals

            for dr, dc in directions:
                count = 1  # Include the current token
                # Check forward direction
                r, c = row + dr, col + dc
                while 0 <= r < rows and 0 <= c < cols and (self.board[r, c] == token or self.board[r, c] is None):
                    count += 1
                    r += dr
                    c += dc
                
                # Check backward direction
                r, c = row - dr, col - dc
                while 0 <= r < rows and 0 <= c < cols and (self.board[r, c] == token or self.board[r, c] is None):
                    count += 1
                    r -= dr
                    c -= dc
                
                # If the token could be part of a line of 4 or more, it's relevant
                if count >= 4:
                    return True
            
            return False

        # Construct the generalized board ID
        id = ''.join(
            str(self.board[row, col]) if is_relevant_token(row, col)
            else ('0' if self.board[row, col] is not None else '.')
            for row in range(rows)
            for col in range(cols)
        )
        return id


    def print(self):
        '''
        Print the current state of the game board.
        '''
        for row in self.board:
            print(" | ".join([str(cell) if cell is not None else ' ' for cell in row]))
            print("-" * (self.cols * 4 - 1))  # Adjust this to create a proper separator