
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

        self.__init_node_list()

    def __init_node_list(self):
        self.current_node = self.get_node()
        self.move_history.append(self.current_node)

    @classmethod
    def from_model_attributes(cls, node_list, model_params):
        """
        Alternative constructor to initialize C4Model with an existing node list and model parameters.
        """
        # Extract rows and columns from model parameters
        rows = model_params.get("rows")
        cols = model_params.get("cols")

        # Create a new C4Model instance
        instance = cls(rows, cols)

        # Set attributes based on provided data
        instance.node_list = node_list
        instance.move_history.pop()
        instance.current_node = node_list[instance.board.generate_general_board_id()]
        instance.move_history.append(instance.current_node)

        instance.made_nodes = model_params.get("made_nodes", len(node_list))

        return instance

    def __make_node(self, board_id = None):
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
        legal_moves = [i for i, is_legal in enumerate(self.current_node.legal_moves) if is_legal]

        # Check if there are any valid legal moves
        if not legal_moves:
            return None

        selected_move = random.choices(legal_moves, k=1)[0]
        return selected_move
    
    def explore_ai(self):
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
        
        self.push_move(maximizingPlayer, col)
        
        legal_moves = [i for i, is_legal in enumerate(self.current_node.legal_moves) if is_legal]

        if depth == 0 or self.tie_detected() or self.win_detected(col):
            #print(self.current_node.legal_moves)
            evaluation = self.current_node.move_weights[self.best_ai()]
            #print("eval at depth ", depth, ":", evaluation)
            return evaluation
        
        if maximizingPlayer == 1:
            self.train(self.explore_ai, 25, 10, 2)
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
            self.train(self.explore_ai, 25, 10, 1)
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
        legal_moves = [i for i, is_legal in enumerate(self.current_node.legal_moves) if is_legal]

        # Check if there are any valid legal moves
        if not legal_moves:
            return None

        # Select the move with the highest weight from the legal moves
        selected_move = max(legal_moves, key=lambda move: self.current_node.move_weights[move])
        return selected_move

    def push_move(self, token, col):
        if col is None:
            return False
        if col < 0 or col >= self.board.cols:
            print("Column out of bounds!")
            return False
        if self.board.drop_token(token, col):
            self.current_node.last_used_move = col
            self.current_node = self.get_node(self.board.generate_general_board_id())
            self.move_history.append(self.current_node)
            self.col_list.append(col)
            return True
        else:
            print("error dropping token")
            return False

    def player_move(self):
        col = int(input("Enter column to drop token in: "))
        return col

    def pop_move(self):
        if len(self.move_history) < 2:
            print("No tokens to pop!")
            return False
        popped_node = self.move_history.pop()
        self.current_node = self.move_history[-1]
        self.board.remove_token(self.current_node.last_used_move)
        self.col_list.pop()
        return popped_node

    def switch_token(self, token):
        
        t = (token) % 2 + 1
        return t

    def train(self, function, lines, iterations, starting_token):
        token = starting_token
        node_after = None
        root_node = self.current_node
        for _ in range(lines):
            its = 0
            while root_node.fully_explored() is False and its < iterations:
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
                    if self.current_node.fully_explored():
                        # Not a win or a tie so the function isnt broken but already explored so 
                        # function returns None type, need this case to solve
                        node_after = self.current_node
                        break

                    
                    move = function()

                last_col = self.col_list[-1]
                while self.current_node != root_node and self.current_node.fully_explored() and self.pop_move():
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

            
        # Push move until win/loss/tie
        # Adjust weight
        # Pop move 

    def tie_detected(self):
        return all(not item for item in self.current_node.legal_moves)
    
    def win_detected(self, col):
        if self.current_node.win_found:
            return True

        row = None
        for r in range(self.board.rows - 1, -1, -1):
            if self.board.board[r][col] is not None:
                row = r

        if row is None:
            print("Error finding token in win_detected")

        win_found = (
            self.__check_line(row, col, 1, 0) or  # Horizontal
            self.__check_line(row, col, 0, 1) or  # Vertical
            self.__check_line(row, col, 1, 1) or  # Diagonal /
            self.__check_line(row, col, 1, -1)    # Diagonal \
        )
        
        if win_found:
            self.current_node.win_found = True  # Mark the node as a winning state
        return win_found

    
    def __check_line(self, row, col, delta_row, delta_col):
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
        self.current_node.print_move_explored()
        self.current_node.print_move_weights()
        
        self.board.print()


    def get_attributes(self):
        return {
            "cols": self.board.cols,
            "rows": self.board.rows,
            "total_nodes": self.made_nodes
        }
    
    def __str__(self):
        return f"rows:{self.board.rows} cols: {self.board.cols} - nodes: {self.made_nodes}"
