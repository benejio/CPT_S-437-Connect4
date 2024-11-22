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
        self.next_move_keys = [None for _ in range(cols)]
        self.explored = [False for _ in range(cols)]
        self.lastmove = None
        self.visits = 0     # Number of times this node has been visited
        self.player = player # either 1 or 2
        self.win = None # In this state, has a player already won?

        self.__initial_move_weights(legal_moves=legal_moves)

    def fully_explored(self):
        """Checks all cols that are leagal moves and have been explored"""
        return all(explored for explored, legal in zip(self.explored, self.legal_moves) if legal)


    def display_weights(self):
        print("Move Weights:")
        for i, weight in enumerate(self.move_weights):
            if self.legal_moves[i]:
                print(f"Column {i}: {weight}")
            else:
                print(f"Column {i}: (Illegal move)")


    def update_next_key(self, col, key):
        self.next_move_keys[col] = key
    
    def get_next_key(self, col):
        return self.next_move_keys[col]
    
    def __initial_move_weights(self, legal_moves):
        '''
            legal_moves an array of booleans with size = cols
            Initiates moves to 0.5 if legal and 0 if illegal
        '''
        for col, is_legal in enumerate(legal_moves):
            self.move_weights[col] = 0.50 if is_legal else 0
        


    def last_move(self, move):
        """Adds a child node to this node."""
        self.lastmove = move

    def increment_visits(self):
        self.visits += 1

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

    def adjust_weights_fuzzy(self, winner, depth, rate, debug = False, last_node = None):

        old_weight = self.move_weights[self.lastmove]

        if (depth == 1):
            self.explored[self.lastmove] = True
            if debug:
                    print("Set node:", self.id, "explored to: ", True)
            if (self.player == winner):
                self.move_weights[self.lastmove] = 1 # winning move!
            elif(winner == 0): # Tie case
                self.move_weights[self.lastmove] = 0.5 # meh move
            else:
                print("Something bad happened!")
        else: # backprop moves
            if (last_node.fully_explored()):
                self.explored[self.lastmove] = True
                if debug:
                    print("Set node:", self.id, "explored to: ", True)
            value = ((last_node.get_best_move() + last_node.get_average_move())) / 2.0
            # print("Last_node.get_best_move() = ", last_node.get_best_move())
            self.move_weights[self.lastmove] = 1 - value


        if (self.player == winner):
            result = "win"
        elif (winner == 0):
            result = "tie"
        else:
            result = "loss"

        if(debug):
            print(
                f"Node ID: {self.id}, "
                f"Move: {self.lastmove}, "
                f"Result: {result}, "
                f"Old Weight: {old_weight:.3f}, "
                f"Adjusted Weight: {self.move_weights[self.lastmove]:.3f}"
            )

    def adjust_weights(self, winner, depth, rate, debug = False, last_node = None):

        old_weight = self.move_weights[self.lastmove]
        
        if (depth == 1):
            if (self.player == winner):
                self.move_weights[self.lastmove] = 1 # winning move!
            elif(winner == 0): # Tie case
                self.move_weights[self.lastmove] = 0.25 # meh move
            else:
                print("Something bad happened!")
        else: # backprop moves
            # print("Last_node.get_best_move() = ", last_node.get_best_move())
            if (last_node.get_best_move() == 0.25):
                self.move_weights[self.lastmove] = .25
            elif (last_node.get_best_move() == 0):
                self.move_weights[self.lastmove] = 0.9
            else:
                self.move_weights[self.lastmove] = 1 - last_node.get_best_move()

        if (self.player == winner):
            result = "win"
        elif (winner == 0):
            result = "tie"
        else:
            result = "loss"

        if(debug):
            print(
                f"Node ID: {self.id}, "
                f"Move: {self.lastmove}, "
                f"Result: {result}, "
                f"Old Weight: {old_weight:.3f}, "
                f"Adjusted Weight: {self.move_weights[self.lastmove]:.3f}"
            )
            