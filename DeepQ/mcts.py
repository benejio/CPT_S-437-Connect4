# mcts.py

import random
import numpy as np
import torch
from Connect4 import Connect4, ROWS, COLUMNS 
from network import Connect4Net

class Node:
    def __init__(self, state, parent=None):
        self.state = state  # Connect4 game state
        self.parent = parent
        self.children = {}  # Dictionary of move -> Node
        self.visits = 0
        self.value = 0

    def is_leaf(self):
        # A node is a leaf if it has no children
        return len(self.children) == 0

    def expand(self, valid_moves):
        """Expands the node by adding all valid moves as children."""
        for move in valid_moves:
            if move not in self.children:
                new_state = Connect4()
                new_state.board = self.state.board.copy()
                new_state.current_player = self.state.current_player
                new_state.make_move(move)
                self.children[move] = Node(new_state, parent=self)

    def best_child(self, c_param=1.4):
        """Selects the best child using the UCT formula."""
        choices_weights = [
            (child.value / (child.visits + 1)) + c_param * np.sqrt(np.log(self.visits + 1) / (child.visits + 1))
            for child in self.children.values()
        ]
        best_move = list(self.children.keys())[np.argmax(choices_weights)]
        return self.children[best_move]

def mcts_search(root, n_iter, model, device):
    for _ in range(n_iter):
        node = root
        # Selection phase
        while not node.is_leaf():
            node = node.best_child()

        # Expansion phase
        valid_moves = [col for col in range(COLUMNS) if node.state.is_valid_location(col)]
        if valid_moves and not node.state.check_winner():
            node.expand(valid_moves)
            node = random.choice(list(node.children.values()))  # Randomly choose a child for simulation

        # Simulation phase (DQN model)
        state_tensor = torch.tensor(node.state.board.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor).cpu().numpy()  # Move to CPU if you need it in numpy format
        best_q_value = np.max(q_values)

        # Backpropagation phase
        while node is not None:
            node.visits += 1
            node.value += best_q_value
            node = node.parent

    # Choose the move with the most visits as the best move
    best_move = max(root.children.items(), key=lambda child: child[1].visits)[0]
    return best_move
