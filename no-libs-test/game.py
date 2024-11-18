import random
from gameboard import GameBoard
from node import Node

class Game:

    def __init__(self, rows, cols):
        self.board = GameBoard(rows, cols)
        self.node_count = 0
        self.current_player = 1
        self.node_map = {}  # Dictionary to store nodes by their unique ID
        self.current_node = self.__get_or_create_node(self.board.board_to_key())
        self.history = []
        self.history.append(self.current_node)
        self.wins = [0,0,0]
        self.games = 0
        
        
        print("New Game Initiated")
    
    def get_node_count(self):
        return self.node_count
    
    def increment_node_count(self):
        self.node_count += 1

    def __get_or_create_node(self, key):
        """
        Retrieves an existing node by its key or creates a new one if it doesn't exist.
        
        Args:
            key (str): The unique identifier for the node.
        
        Returns:
            Node: The corresponding node object.
        """
        if key not in self.node_map:
            self.node_map[key] = Node(id=key, cols=self.board.cols, player=self.current_player, legal_moves = self.get_legal_moves())
            self.increment_node_count()
        return self.node_map[key]
    
    def get_legal_moves(self):
        return [self.board.is_column_available(col) for col in range(self.board.cols)]

    def __switch_current_player(self):
        if self.current_player == 1:
            self.current_player = 2
        elif self.current_player == 2:
            self.current_player = 1

    def ai_move(self):
        # Filter weights based on legal moves
        valid_weights = [
            max(0, min(self.current_node.move_weights[i], 1)) if self.current_node.legal_moves[i] else 0
            for i in range(len(self.current_node.move_weights))
        ]

        # Check if there are any valid moves
        total_weight = sum(valid_weights)
        if total_weight == 0:
            # No legal moves, declare a tie
            return None

        # Normalize the weights so they sum to 1 (probability distribution)
        normalized_weights = [weight / total_weight for weight in valid_weights]

        # Select a move using the normalized weights
        selected_move = random.choices(range(len(self.current_node.move_weights)), weights=normalized_weights, k=1)[0]
        
        return selected_move

    def make_move(self, col, debug = False):
        """
        Makes a move on the board and updates the game state.
        
        Args:
            col (int): The column where the player wants to drop a disc.
            player (int): The player making the move (1 or 2).
        
        Returns:
            bool: True if the move was successful, False otherwise.
        """
        if col == None:
            self.end_game(winner=0, debug=debug)
            return False
        old_key = self.board.board_to_key()
        if self.board.drop_disc(col, self.current_player):
            self.node_map[old_key].last_move(col)
            self.node_map[old_key].increment_visits()
            # Update the game state
            new_state = self.board.board_to_key()
            # Retrieve or create the node for the new state
            self.__switch_current_player()
            self.current_node = self.__get_or_create_node(new_state)
            if self.check_for_win():
                self.end_game(self.current_player, debug=debug)
                return False
            # Track the node in the history
            self.history.append(self.current_node)
            
            return True
        return False
    
    def check_for_win(self):
        """
        Checks if a win condition is met on the game board based on the last dropped token.

        Returns:
            bool: True if a win has occurred, False otherwise.
        """
        if self.board.last_dropped_token_row is None:
            return False  # No moves have been made yet
        if self.current_node.win: # avoid calculating again for same state
            return True

        # Get the last dropped token's position
        row = self.board.last_dropped_token_row
        col = self.board.last_dropped_token_col
        player = self.board.board[row][col]

        if player == 0:
            return False  # Invalid state if the last token is empty
        
        win_found = (
            self._check_line(row, col, 1, 0) or  # Horizontal
            self._check_line(row, col, 0, 1) or  # Vertical
            self._check_line(row, col, 1, 1) or  # Diagonal /
            self._check_line(row, col, 1, -1)    # Diagonal \
        )
        if win_found:
            self.current_node.win = True  # Mark the node as a winning state
        return win_found

    def _check_line(self, row, col, delta_row, delta_col):
        # Check a line for 4 consecutive same-player pieces
        piece = self.board.board[row][col]
        count = 1

        # Check in one direction
        r, c = row + delta_row, col + delta_col
        while 0 <= r < self.board.rows and 0 <= c < self.board.cols and self.board.board[r][c] == piece:
            count += 1
            r += delta_row
            c += delta_col

        # Check in the opposite direction
        r, c = row - delta_row, col - delta_col
        while 0 <= r < self.board.rows and 0 <= c < self.board.cols and self.board.board[r][c] == piece:
            count += 1
            r -= delta_row
            c -= delta_col

        return count >= 4
    
    def end_game(self, winner, debug=False):
        self.games += 1
        if winner == 0: # No winner or Tie
            if debug:
                print("Tie game!")
            self.adjust_weights_for_tie(learning_rate=0.5, debug=debug)
            self.wins[0] += 1
        elif winner == 1:
            self.adjust_weights(2, 0.5, debug=debug)
            if debug:
                print("Player 2 wins")
            self.wins[2] += 1
            
        else:
            self.adjust_weights(1, 0.5, debug=debug)
            if debug:
                print("Player 1 wins")
            self.wins[1] += 1
        return 0
    
    def adjust_weights_for_tie(self, learning_rate, debug = False):
        """
        Adjusts move weights based on the game's history.
        """
        if(debug):
            print("Adjusting weights based on game history...")
        depth = 1
        for node in reversed(self.history):  # Traverse history in reverse order
            if node.lastmove is not None:
                node.adjust_weights_for_tie( depth, learning_rate, debug)
            depth += 1

    def adjust_weights(self, winner, learning_rate, debug=False):
        """
        Adjusts move weights based on the game's history.
        """
        if(debug):
            print("Adjusting weights based on game history...")
        depth = 1
        for node in reversed(self.history):  # Traverse history in reverse order
            if node.lastmove is not None:
                node.adjust_weights(winner, depth, learning_rate, debug)
            depth += 1
    
    def reset_game(self):
        """
        Resets the game board and state for a new game.
        """
        
        self.board = GameBoard(self.board.rows, self.board.cols)
        self.current_node = self.__get_or_create_node(self.board.board_to_key())
        self.history = []
        self.current_player = 1

    def display(self):
        self.board.display()

    def display_wins(self):
        print("Total Games: ", self.games)
        print("Player 1 wins: ", self.wins[1])
        print("Player 2 wins: ", self.wins[2])
        print("Ties wins: ", self.wins[0])
    
    def reset_wins(self):
        self.games = 0
        self.wins = [0,0,0]
