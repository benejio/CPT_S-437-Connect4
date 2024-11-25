# network.py

import torch
import torch.nn as nn
from Connect4 import ROWS, COLUMNS 

class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(ROWS * COLUMNS, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, COLUMNS)

    def forward(self, x):
        x = x.view(-1, ROWS * COLUMNS)  # Flatten the board state
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x