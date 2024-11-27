# mcts.py

import random
import numpy as np
import torch
from Connect4 import Connect4, ROWS, COLUMNS 
from network import Connect4Net

class Node:
    def __init__(self, state, parent=None):
        self.state = state      # Connect4 game state for this node
        self.parent = parent    # Parent node (None for the root)
        self.children = {}      # Dictionary mapping moves to child nodes
        self.visits = 0         # Number of times this node has been visited
        self.value = 0          # Total value accumulated from simulations

    def is_leaf(self):
        return len(self.children) == 0  # A node is a leaf if it has no children

    def expand(self, valid_moves):
        """Expands the node by adding all valid moves as children."""
        for move in valid_moves:
            if move not in self.children:                               # Only expand if the move hasn't been added
                new_state = Connect4()                                  # Create a new Connect4 game state
                new_state.board = self.state.board.copy()               # Copy the current board state
                new_state.current_player = self.state.current_player    # Copy the current player
                new_state.make_move(move)                               # Apply the move to the new state
                self.children[move] = Node(new_state, parent=self)      # Add the new node as a child


    def best_child(self, c_param=1.4):
        """Selects the best child using the UCT formula."""
        choices_weights = [                                                                                         # Calculate UCT (Upper Confidence Bound) score for each child
            (child.value / (child.visits + 1)) + c_param * np.sqrt(np.log(self.visits + 1) / (child.visits + 1))
            for child in self.children.values()
        ]
        best_move = list(self.children.keys())[np.argmax(choices_weights)]                                          # Find the move with the highest UCT score
        return self.children[best_move]                                                                             # Return the child node corresponding to the best move

def mcts_search(root, n_iter, model, device):
    for _ in range(n_iter):     # Perform multiple MCTS iterations
        node = root             # Start at the root node

        # Selection phase
        while not node.is_leaf():       # Traverse down the tree to a leaf node
            node = node.best_child()    # Select the best child node based on UCT formula

        # Expansion phase
        valid_moves = [col for col in range(COLUMNS) if node.state.is_valid_location(col)]      # Find valid moves
        if valid_moves and not node.state.check_winner():                                       # Expand if there are valid moves and no winner yet
            node.expand(valid_moves)                                                            # Add child nodes for all valid moves
            node = random.choice(list(node.children.values()))                                  # Randomly choose a child for simulation

        # Simulation phase (DQN model)
        state_tensor = torch.tensor(node.state.board.flatten(), dtype=torch.float32).unsqueeze(0).to(device)    # Flatten and move board state to GPU
        with torch.no_grad():                                                                                   # Disable gradient computation for efficiency
            q_values = model(state_tensor).cpu().numpy()                                                        # Get Q-values from the DQN model and move to CPU
        best_q_value = np.max(q_values)                                                                         # Get the highest Q-value as the result of the simulation

        # Backpropagation phase
        while node is not None:         # Backpropagate from the simulated node to the root
            node.visits += 1            # Increment the visit count for the node
            node.value += best_q_value  # Update the node's value with the Q-value
            node = node.parent          # Move up to the parent node

    # Choose the move with the most visits as the best move
    best_move = max(root.children.items(), key=lambda child: child[1].visits)[0]  # Select the move with the highest visit count
    return best_move 
