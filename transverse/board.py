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

    def make_node(self):
        return 

    '''
        Returns the row that is next in drop order
    '''
    def __find_clear_space(self, col):
        
        if self.column_heights[col] < 0:  # If column is full
            return None
        return self.column_heights[col]

    '''
        Drops a token at the specified column, 
            if the column is full or column is invalid returns False
    '''
    def drop_token(self, token, col):
        
        if col < 0 or col >= self.cols:
            return False
        row = self.__find_clear_space(col)
        if row:
            self.board[row][col] = token
            self.column_heights[col] -= 1
            return True
        return False 

    '''
        Helper function for generate_id(...)
        Returns true if the token at row, col can be used 
            in a future connect 4
    '''
    def __is_token_relevant(self, row, col):
        token = self.board[row][col]



    def generate_id(self):
        pass

