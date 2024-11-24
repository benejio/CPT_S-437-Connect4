
class Node:
    '''
    
    
    '''

    def __init__(self, board, id):
        self.move_weights = [0.5 for _ in range(board.cols)]
        self.legal_moves = self.__init_legal_moves(board)
        self.explored = [0 for _ in range(board.cols)]
        self.id = id
        pass

    def __init_legal_moves(self, board):
        moves = [True for _ in range(board.cols)]
        for i in range(board.cols):
            if (board.board[0][i] is not None):
                moves[i] = False
        return moves

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