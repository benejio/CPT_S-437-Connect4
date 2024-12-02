import random
import math
from board import Board
from node import Node

class C4Model:

    def __init__(self, rows, cols):

        self.board = Board(rows, cols)
        self.node_list = {} # Dict of all nodes that make up the model
        self.move_history = [] # History of nodes in current board state
        self.current_node = None # current node
        self.col_list = [] # list of column numbers that lead to current board state
        self.made_nodes = 0 # number of nodes in node_list

        self.current_node = self.get_node()
        self.move_history.append(self.current_node)
        
    @classmethod
    def from_model_attributes(cls, node_list, model_params):
        '''
            Constructor to initialize C4Model with an existing node list and model parameters.
        '''
        rows = model_params.get("rows")
        cols = model_params.get("cols")

        instance = cls(rows, cols)

        instance.node_list = node_list
        instance.move_history.pop()

        # Set currentnode and history correctly
        instance.current_node = node_list[instance.board.generate_general_board_id()]
        instance.move_history.append(instance.current_node)

        instance.made_nodes = model_params.get("made_nodes", len(node_list))

        return instance

    def __make_node(self, board_id = None):
        '''
            Makes a new node from board_id
        '''
        if board_id is not None:
            id = board_id
        else:
            id = self.board.generate_general_board_id()
        new_node = Node.from_board(self.board, id)
        self.node_list[id] = new_node
        self.made_nodes +=1
        return self.node_list[id]
    
    def get_node(self, board_id = None):
        '''
            Return node from dict or make new node if it doesnt exist
        '''
        if board_id in self.node_list:
            return self.node_list[board_id]
        return self.node_list.get(board_id, self.__make_node(board_id))




    def random_ai(self):
        '''
            Picks moves randomly
        '''
        legal_moves = [i for i, is_legal in enumerate(self.current_node.legal_moves) if is_legal]

        # Check if there are any valid legal moves
        if not legal_moves:
            return None

        selected_move = random.choices(legal_moves, k=1)[0]
        return selected_move
    
    def explore_ai(self):
        '''
            Explores moves that have yet to be explored
        '''
        legal_moves = [
            i for i, (is_legal, is_explored) 
            in enumerate(zip(self.current_node.legal_moves, self.current_node.explored)) 
            if is_legal and not is_explored
        ]

        # Check if there are any valid legal moves
        if not legal_moves:
            return None

        
        selected_move = random.choices(legal_moves, k=1)[0]
        return selected_move
    
    def minimax_ai(self, starting_token):
        '''
            Not used in final project
            Attempt at implementing alpha-beta pruning and minimax to better calculated states
        '''
        legal_moves = [i for i, is_legal in enumerate(self.current_node.legal_moves) if is_legal]
        move_weights = []

        # Check if there are any valid legal moves
        if not legal_moves:
            return None
        
        for move in legal_moves:
            move_weights.append(self.minimax(4, move, alpha=-math.inf, beta=math.inf, maximizingPlayer=starting_token))
            self.pop_move()
        
        best_move = max(zip(legal_moves, move_weights), key=lambda x: x[1])[0]

        return best_move

        
    def minimax(self, depth, col, alpha, beta, maximizingPlayer):
        '''
            Not used in final project
            Attempt at implementing alpha-beta pruning and minimax to better calculated states
        '''
        
        self.push_move(maximizingPlayer, col)
        
        legal_moves = [i for i, is_legal in enumerate(self.current_node.legal_moves) if is_legal]

        if depth == 0 or self.tie_detected() or self.win_detected(col):
            #print(self.current_node.legal_moves)
            evaluation = self.current_node.move_weights[self.best_ai()]
            #print("eval at depth ", depth, ":", evaluation)
            return evaluation
        
        if maximizingPlayer == 1:
            self.train(self.explore_ai, 25, 10, 2) # Shortly train at each step to get general idea of state
            maxEval = -math.inf
            for move in legal_moves:
                eval = self.minimax(depth-1, move, alpha, beta, 2)
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                self.pop_move()
                if beta <= alpha:
                    #print("Beta less than alpha")
                    break;
            if col == -1:
                print("original thread eval:", maxEval)
            return maxEval
        else:
            self.train(self.explore_ai, 25, 10, 1) # Shortly train at each step to get general idea of state
            minEval = math.inf
            for move in legal_moves:
                eval = self.minimax(depth-1, move, alpha, beta, 1)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                self.pop_move()
                if beta <= alpha:
                    #print("Beta less than alpha")
                    break;
            return minEval
                

    def best_ai(self):
        '''
            Picks the 'best' move according the move_weights for legal_moves
        '''
        legal_moves = [i for i, is_legal in enumerate(self.current_node.legal_moves) if is_legal]

        # Check if there are any valid legal moves
        if not legal_moves:
            return None

        # Select the move with the highest weight from the legal moves
        selected_move = max(legal_moves, key=lambda move: self.current_node.move_weights[move])
        return selected_move

    def player_move(self):
        '''
            Asks the player where to drop a token 
        '''
        col = int(input("Enter column to drop token in: "))
        return col

    def push_move(self, token, col):
        '''
            Pushes a move into move history. 
        '''
        # Validate col
        if col is None:
            return False
        if col < 0 or col >= self.board.cols:
            print("Column out of bounds!")
            return False
        
        # Drop the token into position and set data
        if self.board.drop_token(token, col):
            self.current_node.last_used_move = col
            self.current_node = self.get_node(self.board.generate_general_board_id())
            self.move_history.append(self.current_node)
            self.col_list.append(col)
            return True
        else:
            print("error dropping token")
            return False


    def pop_move(self):
        '''
            Pops the last move from history, undoes the action of push move
        '''
        if len(self.move_history) < 2: # < 2 because the first node should never be popped
            print("No tokens to pop!")
            return False
        popped_node = self.move_history.pop()
        self.current_node = self.move_history[-1]
        self.board.remove_token(self.current_node.last_used_move)
        self.col_list.pop()
        return popped_node

    def switch_token(self, token):
        '''
            Switches the current token from 1 to 2 or from 2 to 1
        '''
        t = (token) % 2 + 1
        return t

    def train(self, function, lines, iterations, starting_token):
        '''
            Trains the model/nodes based on simulated games.
        '''
        token = starting_token
        node_after = None
        root_node = self.current_node
        for _ in range(lines):
            its = 0
            while root_node.is_fully_explored() is False and its < iterations:
                its+=1
                move = function()
                while self.push_move(token, move):
                    token = self.switch_token(token)
                    if self.win_detected(self.col_list[-1]):
                        last_col = self.col_list[-1]
                        self.pop_move()
                        
                        self.current_node.adjust_weight_terminal(last_col, ending_type="win")
                        node_after = self.current_node
                        token = self.switch_token(token)
                        break
                    if self.tie_detected():
                        last_col = self.col_list[-1]
                        self.pop_move()
                        self.current_node.adjust_weight_terminal(last_col, ending_type="tie")
                        node_after = self.current_node
                        token = self.switch_token(token)
                        break
                    if self.current_node.is_fully_explored():
                        # Not a win or a tie so the function isnt broken but already explored so 
                        # function returns None type, need this case to solve
                        node_after = self.current_node
                        break

                    
                    move = function()

                last_col = self.col_list[-1]
                while self.current_node != root_node and self.current_node.is_fully_explored() and self.pop_move():
                    self.current_node.adjust_weight(last_col, node_after_move=node_after)
                    node_after = self.current_node
                    token = self.switch_token(token)
                    if len(self.col_list) > 0:
                        last_col = self.col_list[-1]

            if len(self.col_list) > 0:
                last_col = self.col_list[-1]
            while  self.current_node != root_node and self.pop_move():
                self.current_node.adjust_weight(last_col, node_after_move=node_after)
                node_after = self.current_node
                token = self.switch_token(token)
                if len(self.col_list) > 0:
                    last_col = self.col_list[-1]


    def tie_detected(self):
        '''
            Returns True if all moves in legal_moves are False.
            This only happens if the board is completely full
        '''
        return all(not move for move in self.current_node.legal_moves)
    
    def win_detected(self, col):
        '''
            Checks for a win at the topmost token in a column.
            returns True if a win is detected
        '''
        if self.current_node.win_found: # check if this calculation has already been done
            return True

        row = None
        for r in range(self.board.rows - 1, -1, -1):
            if self.board.board[r][col] is not None:
                row = r # assign row to be the row with the most recently played token 

        if row is None:
            print("Error finding token in win_detected")
            return False

        win_found = (
            self.__check_line(row, col, 1, 0) or  # Horizontal
            self.__check_line(row, col, 0, 1) or  # Vertical
            self.__check_line(row, col, 1, 1) or  # Diagonal /
            self.__check_line(row, col, 1, -1)    # Diagonal \
        )
        
        if win_found:
            self.current_node.win_found = True  # Mark the node as a winning
        return win_found

    
    def __check_line(self, row, col, delta_row, delta_col):
        '''
            Checks if a line starting at (row, col) contains a 4-in-a-row
            We only need to check lines of the most recently placed token
            Params
            row: the row of the token to check
            col: the col of the token to check
            delta_row: the direction of the row to move in each step
            delta_col: the direction of the col to move in each step
        '''
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

    def print(self):
        '''
            Prints information about the current model to the consol
        '''
        self.current_node.print_move_explored()
        self.current_node.print_move_weights()
        self.board.print()


    def get_attributes(self):
        '''
            Returns a dict of attributes
            Used in saving the model to a file
        '''
        return {
            "cols": self.board.cols,
            "rows": self.board.rows,
            "total_nodes": self.made_nodes
        }
    

    def __str__(self):
        '''
            Used in displaying the model for the menu
        '''
        return f"rows:{self.board.rows} cols: {self.board.cols} - nodes: {self.made_nodes}"
