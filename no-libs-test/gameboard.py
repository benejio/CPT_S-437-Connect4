class GameBoard:
    '''
        The GameBoard class is the class that handels the gameboard!
    '''

    def __init__(self, rows=6, cols=7):
        '''
        Initialize a Connect 4 board.
        Args:
            rows (int): Number of rows in the board.
            cols (int): Number of columns in the board.
        '''
        self.rows = rows
        self.cols = cols
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.last_dropped_token_row = None
        self.last_dropped_token_col = None
    
    def board_to_key(self):
        """
        Transforms the current board into a unique key for a key-value map.
        Returns:
            str: A string representation of the board, which can be used as a map key.
        """
        # Flatten the board and convert each element to a string
        flattened_board = [str(cell) for row in self.board for cell in row]
        return ''.join(flattened_board)
    
    def display(self):
        """
        Prints the current board in a readable format.
        """
        for row in self.board:
            print(' '.join(str(cell) for cell in row))
    
    def drop_disc(self, col, player):
        """
        Simulates dropping a disc into a column for a player.
        Args:
            col (int): The column index where the disc is dropped.
            player (int): The player making the move (1 or 2).
        Returns:
            bool: True if the disc was successfully dropped, False otherwise.
        """
        if col < 0 or col >= self.cols:
            print("Invalid column.")
            return False
        for row_idx in range(self.rows - 1, -1, -1):  # Iterate from the bottom row to the top
            if self.board[row_idx][col] == 0:
                self.board[row_idx][col] = player
                self.last_dropped_token_row = row_idx  # Correctly record the row and column index
                self.last_dropped_token_col = col 
                return True
        print("Column is full.")
        return False
    
    


    def is_column_available(self, col):
        if col < 0 or col >= self.cols:
            print("Invalid column.")
            return False
        return self.board[0][col] == 0 # only need to check top row
