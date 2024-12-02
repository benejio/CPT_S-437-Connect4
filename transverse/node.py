
class Node:
    '''
    Each Node represents a board state, including the potential legal_moves, move_weights,
    which nodes have been explored during training, if a win has been found and the last used move
    '''

    def __init__(self, legal_moves, explored, move_weights, id, win_found):
        '''
            Main contructor for Node
        '''
        self.legal_moves = legal_moves
        self.explored = explored
        self.move_weights = move_weights
        self.id = id
        self.win_found = win_found
        self.last_used_move = None # Used in model.pop_move() to know which token to remove

    @classmethod
    def from_dict(cls, data):
        '''
            Constructor to create a Node object from a dictionary.
            Used to create nodes from save file.
        '''
        return cls(
            legal_moves=data["legal_moves"],
            explored=data["explored"],
            move_weights=data["move_weights"],
            id=data["id"],
            win_found=data["win_found"],
        )

    @classmethod
    def from_board(cls, board, id):
        '''
            Constructor to create a Node object based on a board.
            Used to create nodes during training.
        '''
        return cls(
            legal_moves=cls.__init_legal_moves(board),
            explored=[0 for _ in range(board.cols)],
            move_weights=[0.5 for _ in range(board.cols)],
            id=id,
            win_found=False,
        )

    @staticmethod
    def __init_legal_moves( board):
        '''
            Init method to setup legal moves for the node
        '''
        moves = [True for _ in range(board.cols)]
        for i in range(board.cols):
            if (board.board[0][i] is not None):
                moves[i] = False
        return moves
    
    def is_fully_explored(self):
        ''' 
            Returns True if all cols that are leagal moves and have been explored
        '''
        return all(explored for explored, legal in zip(self.explored, self.legal_moves) if legal)

    def adjust_weight_terminal(self, col, ending_type="win"):
        '''
            Called when adjusting the weight of a terminal or leaf node (where the move ends the game)
            Parameters:
                col - the col of the weight to update
                ending_type - either "win" or "tie" - determines the weight to set for the node
        '''
        if ending_type is not None:
            self.explored[col] = True
            if ending_type == "win":
                self.move_weights[col] = 1.0
                for i in range(len(self.explored)): 
                    # Not actually explored but we dont need to because another move is a win
                    # helps reduce amount of nodes we need to visit
                    self.explored[i] = True
            if ending_type == "tie":
                self.move_weights[col] = 0.5

    def adjust_weight(self, col, node_after_move):
        '''
            Called when adjusting the weight of a node that is not a leaf node (ending condition)
            Updates the weight of the node at the specified column
            
            Parameter:
                col - the col of the weight to update
                node_after_move - the node object representing the boardstate after the move at col was made.
        '''
        if (node_after_move.is_fully_explored()):
            self.explored[col] = True
        value =  (node_after_move.get_best_move() + node_after_move.get_adjusted_move_weight()) / 2.0
        self.move_weights[col] = 1-value

    def get_adjusted_move_weight(self):
        '''
            Calculates a single double value that represents a nodes potential 'goodness'.
            This is done by calculating average weight of the move given each possible move 
            is as likely as it is good.
        '''
        total_weight = 0
        valid_weights = []
        adjusted_weights = []

        # fill valid_weights with all weights where move is legal
        for i in range(len(self.move_weights)):
            if self.legal_moves[i]:
                valid_weights.append(self.move_weights[i])

        # if  there are no valid weights return 0
        if not valid_weights:
            return 0

        # Adjust the weights by using weight^3. There is no special reason for using pow 3.
        # using higher power makes it simulate a player who is better at the game. Aka more likely to pick better moves
        adjusted_weights = [pow(weight, 3) for weight in valid_weights]

        total_adjusted_weight = sum(adjusted_weights)
        if total_adjusted_weight == 0: # if total weight is 0, gets here when using weight from losing move
            return 0

        # Normalize the weights
        percent = [weight / sum(adjusted_weights) for weight in adjusted_weights]

        # Adds total weight by multiplying the percent chance of picking a move and the weight of the move picked
        total_weight = sum(percent[i] * valid_weights[i] for i in range(len(valid_weights)))

        return total_weight
        

    def get_best_move(self):
        '''
            Returns the move weight with the highest value of all legal moves
        '''
        return max(
            (weight for weight, is_legal in zip(self.move_weights, self.legal_moves) if is_legal),
            default=None
        )
    
    def print_move_weights(self):
        '''
            Prints the move weights list
        '''
        for i, is_legal in enumerate(self.legal_moves):
            if is_legal:
                print(f"W{i}: {self.move_weights[i]}", end=" | ")
            else:
                print(f"W{i}: False", end=" | ")
        print()

    def print_move_explored(self):
        '''
            Prints the move explored list
        '''
        for i, is_legal in enumerate(self.legal_moves):
            if is_legal:
                print(f"E{i}: {self.explored[i]}", end=" | ")
            else:
                print(f"E{i}: False", end=" | ")
        print()

    def to_dict(self):
        '''
            Used for saving the node to file
        '''
        return {
            "legal_moves": self.legal_moves,
            "explored": self.explored,
            "move_weights": self.move_weights,
            "id": self.id,
            "win_found": self.win_found,
        }
    
