# mcts.py

import random
import numpy as np
import torch
from Connect4 import Connect4, ROWS, COLUMNS 
from network import Connect4Net
from concurrent.futures import ThreadPoolExecutor

class Node:
    def __init__(self, state, parent=None):
        self.state = state      # Connect4 game state for this node
        self.parent = parent    # Parent node (None for the root)
        self.children = {}      # Dictionary mapping moves to child nodes
        self.visits = 0         # Number of times this node has been visited
        self.value = 0          # Total value accumulated from simulations

    def is_leaf(self):
        """Checks if the node is a leaf (has no children)."""
        return len(self.children) == 0

    def expand(self, valid_moves):
        """
        Expands the node by creating child nodes for all valid moves.

        Parameters:
        - valid_moves: List of valid column indices for the current game state.
        """
        for move in valid_moves:
            if move not in self.children:                               # Only expand if the move hasn't been added
                new_state = Connect4()                                  # Create a new Connect4 game state
                new_state.board = self.state.board.copy()               # Copy the current board state
                new_state.current_player = self.state.current_player    # Copy the current player
                new_state.make_move(move)                               # Apply the move to the new state
                self.children[move] = Node(new_state, parent=self)      # Add the new node as a child

    def best_child(self, c_param=1.4):
        """
        Selects the best child node using the Upper Confidence Bound (UCT) formula.

        Parameters:
        - c_param: Exploration-exploitation tradeoff parameter.

        Returns:
        - The child node with the highest UCT score.
        """
        choices_weights = [                                                                                        
            (child.value / (child.visits + 1)) + c_param * np.sqrt(np.log(self.visits + 1) / (child.visits + 1))   # Calculate UCT score
            for child in self.children.values()
        ]
        best_move = list(self.children.keys())[np.argmax(choices_weights)]                                         # Find the move with the highest UCT score
        return self.children[best_move]                                                                            # Return the child node corresponding to the best move

def mcts_search(root, n_iter, model, device):
    """
    Performs Monte Carlo Tree Search (MCTS) to determine the best move.

    Parameters:
    - root: The root node representing the current game state.
    - n_iter: Number of MCTS iterations to perform.
    - model: Neural network model for state evaluation.
    - device: Device (CPU/GPU) for running the model.

    Returns:
    - best_move: The move with the highest visit count.
    """
    with ThreadPoolExecutor(max_workers=4) as executor:  # Use multithreading for simulation phase
        for _ in range(n_iter):
            node = root

            # Selection phase: Traverse the tree to find a leaf node
            while not node.is_leaf():
                node = node.best_child()

            # Expansion phase: Expand the leaf node if valid moves exist
            valid_moves = [col for col in range(COLUMNS) if node.state.is_valid_location(col)]
            if valid_moves and not node.state.check_winner():  # Only expand if no winner and valid moves are available
                node.expand(valid_moves)
                node = random.choice(list(node.children.values()))  # Choose a random child for simulation

            # Simulation phase: Perform simulations in parallel
            simulation_results = list(executor.map(lambda _: simulate_game(node, model, device), range(n_iter)))

            # Backpropagation phase: Update the values and visits along the path back to the root
            for result in simulation_results:
                while node is not None:
                    node.visits += 1
                    node.value += result  # Update value based on simulation result
                    node = node.parent

    # Choose the move with the most visits as the best move
    best_move = max(root.children.items(), key=lambda child: child[1].visits)[0]
    return best_move

def simulate_game(node, model, device):
    """
    Simulates a game from the given node using the neural network model.

    Parameters:
    - node: Node to simulate the game from.
    - model: Neural network model to evaluate states.
    - device: Device (CPU/GPU) for running the model.

    Returns:
    - The best Q-value (predicted reward) as the simulation result.
    """
    # Prepare the board state as a tensor for the model
    state_tensor = torch.tensor(node.state.board.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():  # Disable gradient calculation for evaluation
        q_values = model(state_tensor).cpu().numpy()  # Evaluate Q-values on CPU
    return np.max(q_values)  # Return the best Q-value as the simulation result
