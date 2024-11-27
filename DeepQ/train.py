import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Connect4 import Connect4, EMPTY
from network import Connect4Net
from mcts import mcts_search, Node
from torch.nn.parallel import DataParallel as DP

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU


#batch   # The number of training examples (experiences) used in one training iteration. 
         # Larger batch sizes can improve stability but require more memory.

#gamma   # The discount factor for future rewards in reinforcement learning. 
         # A value close to 1 means the model values future rewards almost as much as immediate rewards.

def train_dqn(model, optimizer, replay_buffer, batch_size, gamma=0.995, step=0):
    if len(replay_buffer) < batch_size:
        return None  # Not enough experiences in replay buffer to train

    # Sample a batch of experiences
    batch = random.sample(replay_buffer, batch_size)            # Randomly sample a batch from the replay buffer
    states, actions, rewards, next_states, dones = zip(*batch)  # Unpack the batch into individual components

    # Convert to tensors and move to device
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)                 # Convert states to tensor
    actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(device)    # Convert actions to tensor
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)               # Convert rewards to tensor
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)       # Convert next states to tensor
    dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)                   # Convert done flags to tensor

    # Compute Q-values for current states
    q_values = model(states).gather(1, actions).squeeze()  # Get Q-values for taken actions

    # Compute target Q-values for next states
    next_q_values = model(next_states).max(1)[0]                        # Max Q-value for the next states
    target_q_values = rewards + (1 - dones) * gamma * next_q_values     # Bellman equation for target Q-values

    # Calculate the loss
    loss = F.mse_loss(q_values, target_q_values.detach())  # Use Mean Squared Error loss

    # Optimize the model
    optimizer.zero_grad()                                               # Clear previous gradients
    loss.backward()                                                     # Backpropagate the loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # Clip gradients to prevent explosion
    optimizer.step()                                                    # Update the model parameters

    return loss.item()  # Return the loss value

def self_play_game(model, replay_buffer, game_num, n_iter=5000):
    game = Connect4()           # Initialize a new Connect4 game
    game.reset()                # Reset the game state
    state = game.board.copy()   # Get the initial board state
    done = False                # Initialize the game completion flag
    move_count = 0              # Track the number of moves made

    while not done:
        root = Node(game)                                                       # Create the root node with the current game state
        best_move = mcts_search(root, n_iter=n_iter, model=model, device=device)  # Perform MCTS to find the best move

        # Perform the move in the game
        game.make_move(best_move)       # Update the game state with the selected move
        new_state = game.board.copy()   # Get the updated board state

        # Assign reward and done flag
        if game.check_winner():                                         # Check if the game has a winner
            reward = 1 if game.current_player == 1 else -1              # Reward for the current player
            done = True                                                 # Mark the game as finished
        else:
            reward = 0                          # Neutral reward for non-terminal states
            done = np.all(game.board != EMPTY)  # Check if the board is full (draw)

        # Append to replay buffer for both players
        replay_buffer.append((state.flatten(), best_move, reward, new_state.flatten(), done))  # Store current player's experience
        replay_buffer.append((-state.flatten(), best_move, -reward, -new_state.flatten(), done))  # Store opponent's experience

        # Switch player
        game.switch_player()  # Change the current player

        # Update current state
        state = new_state   # Update the current state
        move_count += 1     # Increment the move count

def dynamic_n_iter(game_num, total_games, min_iter=100, max_iter=10000):
    """
    Dynamically calculate the number of MCTS iterations based on game progress.

    Parameters:
    - game_num: Current game number.
    - total_games: Total number of games to be played.
    - min_iter: Minimum iterations for early games.
    - max_iter: Maximum iterations for later games.

    Returns:
    - n_iter: Scaled number of iterations.
    """
    # Linearly interpolate n_iter based on game progress
    progress = game_num / total_games
    n_iter = int(min_iter + (max_iter - min_iter) * progress)
    return n_iter

def load_checkpoint(file_path, model, optimizer):
    if os.path.isfile(file_path):                                       # Check if the checkpoint file exists
        checkpoint = torch.load(file_path)                              # Load the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])           # Load the model state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   # Load the optimizer state
        replay_buffer = checkpoint.get('replay_buffer', [])             # Load the replay buffer (if available)
        game_num = checkpoint.get('game_num', 0)                        # Load the game number (if available)
        return replay_buffer, game_num                                  # Return the replay buffer and game number
    else:                                                               # else
        return [], 0                                                    # Return an empty replay buffer and start from game 0 if no checkpoint is found

def save_checkpoint(model, optimizer, replay_buffer, game_num, file_path="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),             # Save the model parameters
        'optimizer_state_dict': optimizer.state_dict(),     # Save the optimizer state
        'replay_buffer': replay_buffer,                     # Save the replay buffer
        'game_num': game_num                                # Save the current game number
    }
    torch.save(checkpoint, file_path)                       # Save the checkpoint to file

def main():
    model = Connect4Net().to(device)                        # Initialize the Connect4 model and move it to the device
    if torch.cuda.device_count() > 1:
        model = DP(model)                                   # Enable multi-GPU training on a single node                              
    optimizer = optim.Adam(model.parameters(), lr=0.0005)   # Initialize the optimizer with a learning rate of 0.0005
    replay_buffer = []                                      # Initialize an empty replay buffer
    total_games = 50000                                     # Total number of games to be played
    total_steps = 0                                         # Initialize the step counter
    checkpoint_path = "checkpoint.pth"                      # Path to save the checkpoint
    checkpoint_path2 = "checkpoint2.pth"                    # Path to save the checkpoint, iteratively
    batch_size = 256                                        # Initial batch size for training
    gamma = 0.995


    # Load from checkpoint if available
    replay_buffer, start_game = load_checkpoint(checkpoint_path, model, optimizer)  # Load checkpoint data

    # Self-play and training loop starting from last checkpoint
    for game_num in range(start_game, total_games):
        print(f"Starting game {game_num + 1}/{total_games}")    # Display progress
        self_play_game(model, replay_buffer, game_num + 1, dynamic_n_iter(game_num, total_games))      # Perform a self-play game

        try:
            loss = train_dqn(model, optimizer, replay_buffer, batch_size=batch_size, gamma=gamma, step=total_steps)
        except RuntimeError as e:
            if 'out of memory' in str(e):   # Handle GPU out-of-memory errors
                batch_size //= 2            # Halve the batch size and retry
                # print(f"Out of memory. Reducing batch size to {batch_size} and retrying.")
                loss = train_dqn(model, optimizer, replay_buffer, batch_size=batch_size, gamma=gamma, step=total_steps)
            else:
                raise e  # Re-raise other exceptions

        total_steps += 1  # Increment the step counter

        # Periodic updates and checkpoint saving
        if (game_num + 1) % 100 == 0:  # Save checkpoint every 100 games
            save_checkpoint(model, optimizer, replay_buffer, game_num + 1, checkpoint_path2)  # Save progress

        # Periodic updates and checkpoint saving :: Altered for HPC
        if (game_num) == total_games:                                                           # Save checkpoint
            save_checkpoint(model, optimizer, replay_buffer, game_num + 1, checkpoint_path)     # Save progress
            with open(checkpoint_path, 'rb') as f:  # Open the file in binary mode
                binary_data = f.read()  # Read the entire binary content
                print(binary_data)  # Print the raw binary data


if __name__ == "__main__":
    main()  # Run the main function
