
import random
from board import Board
from node import Node

class C4Model:

    def __init__(self, rows, cols):

        self.board = Board(rows, cols)
        self.node_list = {}
        self.move_history = []
        self.look_ahead = []
        self.current_node = None

        self.__init_node_list()


    def __make_node(self, board_id = None):
        if board_id is not None:
            id = board_id
        else:
            id = self.board.generate_id()
        new_node = Node(self.board, id)
        self.node_list[id] = new_node
        return self.node_list[id]
    
    def get_node(self, board_id = None):
        '''
            Return node from dict or make new node if it doesnt exist
        '''
        return self.node_list.get(board_id, self.__make_node(board_id))

    def __init_node_list(self):
        self.current_node = self.get_node()
        self.move_history.append(self.current_node)
        


    def look_ahead(self, node):
        # Stop if node does not exist or no legal moves
        # Call look_ahead on legal moves
        pass


    def random_ai(self):
        legal_moves = [i for i, is_legal in enumerate(self.current_node.legal_moves) if is_legal]

        # Check if there are any valid legal moves
        if not legal_moves:
            return None

        selected_move = random.choices(legal_moves, k=1)[0]
        return selected_move


    def make_move(self, token, col):
        if col is None:
            self.end_game(losser=0)
            return False
        
        self.board.drop_token(token, col)
        self.current_node = self.get_node(self.board.generate_id())
    




    def train(self):


        pass

    

    def print(self):
        self.current_node.print_move_explored()
        self.current_node.print_move_weights()
        
        self.board.print()