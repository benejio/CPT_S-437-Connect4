
class Node:
    '''
    
    
    '''

    def __init__(self, board, id):
        self.move_weights = [0.5 for _ in range(board.cols)]
        self.legal_moves = self.__init_legal_moves(board)
        self.explored = [0 for _ in range(board.cols)]
        self.id = id
        self.last_used_move = None
        self.win_found = False
        pass

    def __init_legal_moves(self, board):
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
            if ending_type == "tie":
                self.move_weights[col] = 0.5

    def adjust_weight(self, col, node_after_move):
        if (node_after_move.fully_explored()):
            self.explored[col] = True
        value = ((3* node_after_move.get_best_move() + node_after_move.get_average_move())) / 4.0
        self.move_weights[col] = 1-value


    def get_best_move(self):
        return max(self.move_weights)
    
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