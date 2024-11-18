class Node:
    '''
        The Node class represents a 'state' of the gameboard.
        The Node has an id to represent the state aswell as the move weights
            that the program will use to select a move.
    '''

    def __init__(self, id, cols, player, legal_moves):
        self.id = id
        self.move_weights = [0.5 for _ in range(cols)] # baseline weights
        self.legal_moves = legal_moves
        self.lastmove = None
        self.visits = 0     # Number of times this node has been visited
        self.player = player # either 1 or 2
        self.win = None # In this state, has a player already won?

        self.__initial_move_weights(legal_moves=legal_moves)


    
    def __initial_move_weights(self, legal_moves):
        '''
            legal_moves an array of booleans with size = cols
            Initiates moves to 0.5 if legal and 0 if illegal
        '''
        for col, is_legal in enumerate(legal_moves):
            self.move_weights[col] = 0.5 if is_legal else 0
        


    def last_move(self, move):
        """Adds a child node to this node."""
        self.lastmove = move

    def increment_visits(self):
        self.visits += 1

    def adjust_weights_for_tie(self, depth, rate, debug = False):
        final_rate = 1.0/(depth*depth) * rate
        old_weight = self.move_weights[self.lastmove]
        result = "tie"
        if self.move_weights[self.lastmove] <= 0.6: # Tie trends toward 0.6
            self.move_weights[self.lastmove] = min(0.6, self.move_weights[self.lastmove]+final_rate)
        else:
            self.move_weights[self.lastmove] = max(0.6, self.move_weights[self.lastmove]-final_rate)
        

        self.move_weights[self.lastmove] = max(0, min(self.move_weights[self.lastmove], 1))
        if(debug):
            print(
                f"Node ID: {self.id}, "
                f"Move: {self.lastmove}, "
                f"Result: {result}, "
                f"Old Weight: {old_weight:.3f}, "
                f"Adjusted Weight: {self.move_weights[self.lastmove]:.3f}"
            )

    def adjust_weights(self, winner, depth, rate, debug = False):
        final_rate = 1.0/(depth*depth) * rate
        old_weight = self.move_weights[self.lastmove]
        if self.player == winner:
            self.move_weights[self.lastmove] += final_rate
            result = "win"
        else:
            self.move_weights[self.lastmove] -= final_rate
            result = "loss"
        

        self.move_weights[self.lastmove] = max(0, min(self.move_weights[self.lastmove], 1))
        if(debug):
            print(
                f"Node ID: {self.id}, "
                f"Move: {self.lastmove}, "
                f"Result: {result}, "
                f"Old Weight: {old_weight:.3f}, "
                f"Adjusted Weight: {self.move_weights[self.lastmove]:.3f}"
            )
            