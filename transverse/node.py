
class Node:
    '''
    
    
    '''

    def __init__(self, board, id):
        self.__move_weights = [None for _ in range(board.cols)]
        self.__legal_moves = self.__init_legal_moves(board)
        self.__id = id
        pass

    def __init_legal_moves(self, board):
        moves = [True for _ in range(board.cols)]
        for i in range(board.cols):
            if (board.board[0][board.cols] != None):
                moves[board.cols] = False
        return moves

    