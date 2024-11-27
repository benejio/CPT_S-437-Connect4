# network.py

import torch
import torch.nn as nn
from Connect4 import ROWS, COLUMNS 

class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()  # Initialize the parent class (nn.Module).

        # First convolutional layer: Converts 1-channel (grayscale board input) to 32 channels.
        # kernel_size=4: 4x4 filters (small square regions of the board).
        # stride=1: Moves the filter 1 cell at a time.
        # padding=2: Adds padding of 2 cells around the edges to preserve spatial dimensions.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=2)

        # Second convolutional layer: Converts 32 channels to 64 channels.
        # Same kernel size, stride, and padding as the first convolutional layer.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2)

        # First fully connected (linear) layer:
        # Input size = 64 * 8 * 9 = flattened output from the convolutional layers.
        # Output size = 128 neurons (hidden layer).
        self.fc1 = nn.Linear(64 * 8 * 9, 128)

        # Second fully connected layer:
        # Input size = 128 neurons (from the previous layer).
        # Output size = 64 neurons (hidden layer).
        self.fc2 = nn.Linear(128, 64)

        # Final fully connected layer:
        # Input size = 64 neurons (from the previous layer).
        # Output size = COLUMNS neurons (number of Connect4 columns = 7).
        # This outputs scores (Q-values) for each column.
        self.fc3 = nn.Linear(64, COLUMNS)


    def forward(self, x):
        # Reshape input into a 4D tensor for the convolutional layers:
        # -1: Dynamic batch size.
        # 1: Single input channel (grayscale board).
        # ROWS: Number of rows in the board.
        # COLUMNS: Number of columns in the board.
        x = x.view(-1, 1, ROWS, COLUMNS)

        # Apply the first convolutional layer followed by ReLU activation:
        # Detects basic patterns or features in the input board.
        x = torch.relu(self.conv1(x))

        # Apply the second convolutional layer followed by ReLU activation:
        # Detects higher-level patterns or features based on the output of the first layer.
        x = torch.relu(self.conv2(x))

        # Flatten the 3D output (batch_size, channels, height, width) into a 2D tensor:
        # (batch_size, flattened_size).
        x = x.view(x.size(0), -1)

        # Pass the flattened data through the first fully connected layer with ReLU activation:
        # Learns high-level abstract features based on the convolutional outputs.
        x = torch.relu(self.fc1(x))

        # Pass through the second fully connected layer with ReLU activation:
        # Further refines the features and maps them closer to the output dimension.
        x = torch.relu(self.fc2(x))

        # Pass through the final fully connected layer (no activation):
        # Outputs Q-values (scores) for each column of the board.
        x = self.fc3(x)

        return x  # Return the Q-values for each column.
