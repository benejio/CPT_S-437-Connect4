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
        self.relevant_board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.last_dropped_token_row = None
        self.last_dropped_token_col = None
    
    
    def board_to_key(self):
        """
        Transforms the current board into a unique key for a key-value map, considering only
        relevant positions that can still contribute to a winning condition.

        Returns:
            str: A string representation of the simplified board, which can be used as a map key.
        """
        rows, cols = len(self.board), len(self.board[0])
        normalized_board = [[0 for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                # Skip empty cells
                if self.board[r][c] == 0:
                    continue
                
                # Check if this cell can contribute to a winning line
                if self._is_relevant(r, c):
                    normalized_board[r][c] = self.board[r][c]
                else:
                    # Replace irrelevant positions with a neutral value (e.g., 9)
                    normalized_board[r][c] = 9
        
        # Flatten the normalized board and convert it to a string key
        flattened_board = [str(cell) for row in normalized_board for cell in row]
        return ''.join(flattened_board)

    def _is_relevant(self, row, col):
        """
        Determines if a token at (row, col) is relevant for potential winning lines.
        A position is relevant if it is part of a line of 4 (or could potentially form one).
        
        Args:
            row (int): Row index of the token.
            col (int): Column index of the token.
        
        Returns:
            bool: True if relevant, False otherwise.
        """

        player = self.board[row][col]
        
        # If the cell is empty or irrelevant to connect-4 possibilities, return False
        if player == 0:
            return False  # Empty cells are not inherently relevant

        rows, cols = len(self.board), len(self.board[0])
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Vertical, horizontal, and diagonals

        for dr, dc in directions:
            count = 1

            # Check in one direction
            r, c = row + dr, col + dc
            while 0 <= r < rows and 0 <= c < cols:
                if self.board[r][c] == player or self.board[r][c] == 0:
                    count += 1
                else:
                    break
                r += dr
                c += dc

            if count >= 4:
                return True
            
            # Check in the opposite direction
            r, c = row - dr, col - dc
            while 0 <= r < rows and 0 <= c < cols:
                if self.board[r][c] == player or self.board[r][c] == 0:
                    count += 1
                else:
                    break
                r -= dr
                c -= dc

            # A line is relevant if it has potential to form a connect-4
            if count >= 4:
                return True

        return False
    

    '''
    def board_to_key(self):
        """
        Transforms the current board into a unique key for a key-value map.
        Returns:
            str: A string representation of the board, which can be used as a map key.
        """
        # Flatten the board and convert each element to a string
        flattened_board = [str(cell) for row in self.board for cell in row]
        return ''.join(flattened_board)
    '''
        
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
    
    def undrop_disc(self, col):
        if col < 0 or col >= self.cols:
            print("Invalid column.")
            return False
        for row_idx in range(self.rows):  # Iterate from the top row to the bottom
            if self.board[row_idx][col] != 0:
                self.board[row_idx][col] = 0
                return True
        print("Column is empty.")
        return False
    
    
    def is_column_available(self, col):
        if col < 0 or col >= self.cols:
            print("Invalid column.")
            return False
        return self.board[0][col] == 0 # only need to check top row

    def board_is_full(self):
        return all(self.board[0][col] != 0 for col in range(self.cols))