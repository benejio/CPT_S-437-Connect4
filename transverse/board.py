import numpy as np
from node import Node
import hashlib

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
        self.node_list = {}

    def __make_node(self):
        id = self.generate_id()
        new_node = Node(self, id)
        self.node_list[id] = new_node
        return self.node_list[id]
    
    def get_node(self, board_id):
        '''
            Return node from dict or make new node if it doesnt exist
        '''
        return self.node_list.get(board_id, self.__make_node())


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
    TODO: Change to optimized version

    Generates an id based on the current board state
    '''
    def generate_id(self):
        encoded_board = np.where(self.board == None, -1, self.board)
        
        board_bytes = encoded_board.tobytes()
        
        board_id = hashlib.sha256(board_bytes).hexdigest()
        
        return board_id

    def print(self):
        '''
        Print the current state of the game board.
        '''
        for row in self.board:
            print(" | ".join([str(cell) if cell is not None else ' ' for cell in row]))
            print("-" * (self.cols * 4 - 1))  # Adjust this to create a proper separator
    
    