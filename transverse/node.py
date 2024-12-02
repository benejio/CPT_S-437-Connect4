
class Node:
    '''
    
    
    '''

    def __init__(self, legal_moves, explored, move_weights, id, win_found):
        """
        Primary constructor for the Node class.
        """
        self.legal_moves = legal_moves
        self.explored = explored
        self.move_weights = move_weights
        self.id = id
        self.win_found = win_found
        self.last_used_move = None

    @classmethod
    def from_dict(cls, data):
        """
        Alternative constructor to reconstruct a Node object from a dictionary.
        """
        return cls(
            legal_moves=data["legal_moves"],
            explored=data["explored"],
            move_weights=data["move_weights"],
            id=data["id"],
            win_found=data["win_found"],
        )

    @classmethod
    def from_board(cls, board, id):
        """
        Alternative constructor to create a Node object based on a board.
        """
        return cls(
            legal_moves=cls.__init_legal_moves(board),
            explored=[0 for _ in range(board.cols)],
            move_weights=[0.5 for _ in range(board.cols)],
            id=id,
            win_found=False,
        )

    @staticmethod
    def __init_legal_moves( board):
        moves = [True for _ in range(board.cols)]
        for i in range(board.cols):
            if (board.board[0][i] is not None):
                moves[i] = False
        return moves
    
    def fully_explored(self):
        """Checks all cols that are leagal moves and have been explored"""
        return all(explored for explored, legal in zip(self.explored, self.legal_moves) if legal)


    def adjust_weight_terminal(self, col, ending_type="win"):
        if ending_type is not None:
            self.explored[col] = True
            if ending_type == "win":
                self.move_weights[col] = 1.0
                for i in range(len(self.explored)): # Not actually explored but we dont need to because another move is a win
                    self.explored[i] = True
            if ending_type == "tie":
                self.move_weights[col] = 0.5

    def adjust_weight(self, col, node_after_move):
        if (node_after_move.fully_explored()):
            self.explored[col] = True
        # value = ((3* node_after_move.get_best_move() + node_after_move.get_average_move())) / 4.0
        value =  (node_after_move.get_best_move() + node_after_move.get_adjusted_move_weight()) / 2.0
        self.move_weights[col] = 1-value

    def get_adjusted_move_weight(self):
        total_weight = 0
        valid_weights = []

        # Gather weights of all legal moves greater than or equal to 0.2
        for i in range(len(self.move_weights)):
            if self.legal_moves[i]:
                valid_weights.append(self.move_weights[i])
        
        adjusted_weights = []

        if not valid_weights:
            return 0

        adjusted_weights = [pow(weight, 3) for weight in valid_weights]

        total_adjusted_weight = sum(adjusted_weights)
        if total_adjusted_weight == 0:
            return 0

        # Normalize the adjusted weights
        percent = [weight / sum(adjusted_weights) for weight in adjusted_weights]

        total_weight = sum(percent[i] * valid_weights[i] for i in range(len(valid_weights)))

        return total_weight
        

    def get_average_move_excluding_low_weights(self):
        total_weight = 0
        valid_weights = []

        # Gather weights of all legal moves greater than or equal to 0.2
        for i in range(len(self.move_weights)):
            if self.legal_moves[i] and self.move_weights[i] >= 0.2:
                valid_weights.append(self.move_weights[i])
        
        # If no valid moves remain, return 0
        if not valid_weights:
            return self.get_average_move_except_worst()

        # Calculate the total weight of valid moves
        total_weight = sum(valid_weights)

        # Calculate and return the average weight
        return total_weight / len(valid_weights)


    def get_average_move_except_worst(self):
        '''
            returns the average weights of all moves except for the worst move
            This is used to simulate a player that will never pick the absolute worst move
        '''
        total_weight = 0
        legal_weights = []

        # Gather weights of all legal moves
        for i in range(len(self.move_weights)):
            if self.legal_moves[i]:
                legal_weights.append(self.move_weights[i])
        
        # Return 0 if there are no legal moves
        if not legal_weights:
            return 0

        # If there's only one legal move, return its weight
        if len(legal_weights) == 1:
            return legal_weights[0]

        # Exclude the worst move (minimum weight)
        worst_move_weight = min(legal_weights)
        total_weight = sum(legal_weights) - worst_move_weight

        # Calculate the average excluding the worst move
        return total_weight / (len(legal_weights) - 1)



    def get_best_move(self):
        return max(
            (weight for weight, is_legal in zip(self.move_weights, self.legal_moves) if is_legal),
            default=None
        )
        
    def get_average_move(self):
        total_weight = 0
        legal_move_count = 0
        
        for i in range(len(self.move_weights)):
            if self.legal_moves[i]:
                total_weight += self.move_weights[i]
                legal_move_count += 1

        # Return 0 if there are no legal moves
        if legal_move_count == 0:
            return 0
        
        # Calculate and return the average weight
        return total_weight / legal_move_count
    
    def print_move_weights(self):
        for i, is_legal in enumerate(self.legal_moves):
            if is_legal:
                print(f"W{i}: {self.move_weights[i]}", end=" | ")
            else:
                print(f"W{i}: False", end=" | ")
        print()

    def print_move_explored(self):
        for i, is_legal in enumerate(self.legal_moves):
            if is_legal:
                print(f"E{i}: {self.explored[i]}", end=" | ")
            else:
                print(f"E{i}: False", end=" | ")
        print()

    def to_dict(self):
        return {
            "legal_moves": self.legal_moves,
            "explored": self.explored,
            "move_weights": self.move_weights,
            "id": self.id,
            "win_found": self.win_found,
        }
    
