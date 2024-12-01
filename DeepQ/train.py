import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Connect4 import Connect4, EMPTY, COLUMNS, ROWS
from network import Connect4Net
from mcts import mcts_search, Node
from torch.nn.parallel import DataParallel as DP
from torch.optim.lr_scheduler import CyclicLR

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU

#batch   # The number of training examples (experiences) used in one training iteration. 
         # Larger batch sizes can improve stability but require more memory.

#gamma   # The discount factor for future rewards in reinforcement learning. 
         # A value close to 1 means the model values future rewards almost as much as immediate rewards.

# Constants for learning rate adjustment
initial_lr = 0.001  # High initial learning rate
min_lr = 1e-8  # Minimum learning rate
lr_decay_factor = 0.5  # Factor to reduce learning rate on loss spike
start_loss_threshold = 1.0  # Initial threshold
min_loss_threshold = 0.0005  # Minimum threshold
step_interval = 500  # Interval for adjusting loss threshold
total_steps = 50000 # Total training steps
warmup_steps = int(0.1 * total_steps)  # Number of steps for warm-up phase
warmup_initial_lr = 1e-5  # Starting learning rate during warm-up

# def get_scheduler(optimizer, total_steps):
#     """
#     Returns a scheduler that decays learning rate gradually.
#     """
#     decay_rate = (min_lr / initial_lr) ** (1 / (total_steps // 100))  # Exponential decay rate
#     return ExponentialLR(optimizer, gamma=decay_rate)

def get_scheduler(optimizer):
    return CyclicLR(optimizer, base_lr=warmup_initial_lr, max_lr=initial_lr, step_size_up=warmup_steps, mode='triangular')


def get_dynamic_loss_threshold(loss_history, factor=0.1, min_threshold=0.0005):
    """
    Dynamically calculate the loss increase threshold based on a rolling average.
    """
    if len(loss_history) < 10:
        return start_loss_threshold  # Use initial threshold for early training
    avg_loss = sum(loss_history[-10:]) / len(loss_history[-10:])  # Rolling average
    return max(avg_loss * factor, min_threshold)

# Dynamic learning rate function
def adjust_learning_rate(optimizer, scheduler, current_loss, loss_history, step):
    """
    Adjust the learning rate based on loss trends or warm-up phase.
    """
    if step < warmup_steps:  # Warm-up phase
        lr = warmup_initial_lr + (initial_lr - warmup_initial_lr) * (step / warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print(f"Warm-up phase: Adjusted learning rate to {lr}")
    else:
        # Calculate the dynamic loss threshold
        loss_threshold = get_dynamic_loss_threshold(loss_history)
        # print(f"Dynamic loss threshold: {loss_threshold}")

        if len(loss_history) > 1 and current_loss > loss_history[-1] + loss_threshold:
            # Loss spike detected, reduce learning rate
            if optimizer.param_groups[0]['lr'] <= min_lr:
                # print("Learning rate has reached minimum; skipping further adjustments.")
                return
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * lr_decay_factor, min_lr * 10)
            # print(f"Adjusted learning rate to {optimizer.param_groups[0]['lr']} due to loss spike.")
        else:
            # Apply scheduler's gradual decay
            optimizer.step()  # Step optimizer first
            scheduler.step()  # Then step the scheduler
            # print(f"Scheduler adjusted learning rate to {optimizer.param_groups[0]['lr']}")

def train_dqn(model, optimizer, scheduler, replay_buffer, batch_size, gamma=0.995, step=0, loss_history=None):
    """Trains the model using experiences from the replay buffer."""
    if len(replay_buffer) < batch_size:
        return None, loss_history  # Not enough experiences in the replay buffer to train

    # Sample a batch of experiences
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

    # Compute Q-values
    q_values = model(states).gather(1, actions).squeeze()
    next_q_values = model(next_states).max(1)[0]
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Loss calculation
    loss = F.mse_loss(q_values, target_q_values.detach())

    # Update loss history
    if loss_history is None:
        loss_history = []
    loss_history.append(loss.item())

    # Adjust learning rate
    adjust_learning_rate(optimizer, scheduler, loss.item(), loss_history, step)

    # Optimize model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Log loss
    # if step % 100 == 0:
    #     print(f"Step {step}: Loss = {loss.item()}, Learning Rate = {optimizer.param_groups[0]['lr']}")


    return loss.item(), loss_history  # Return loss and updated loss history

MAX_REPLAY_BUFFER_SIZE = 100000  # Maximum replay buffer size

def self_play_game(model, replay_buffer, game_num, n_iter=1000):
    """Plays a game of Connect 4 using MCTS and appends experiences to the replay buffer."""
    game = Connect4()           # Initialize a new Connect 4 game
    game.reset()                # Reset the game state
    state = game.board.copy()   # Get the initial board state
    done = False                # Initialize the game completion flag
    move_count = 0              # Track the number of moves made

    while not done:
        # Check for an immediate winning move
        for col in range(COLUMNS):
            if game.is_valid_location(col):                # Ensure the column is valid
                temp_game = Connect4()                     # Create a temporary game instance
                temp_game.board = game.board.copy()        # Copy the current board state
                temp_game.make_move(col)                   # Make the move in the temporary game
                if temp_game.check_winner():               # Check if this move wins the game
                    game.make_move(col)                    # AI takes the winning move
                    done = True                            # Mark the game as finished
                    reward = 1                             # AI wins, so reward = 1
                    #  print(f"Game {game_num}, Move {move_count}: AI wins with column {col}")  # Debugging
                    replay_buffer.append((state.flatten(), col, reward, state.flatten(), done))

                    # Ensure replay buffer doesn't exceed maximum size
                    if len(replay_buffer) > MAX_REPLAY_BUFFER_SIZE:
                        replay_buffer.pop(0)
                    return  # Exit the loop and function

        # Block opponent's winning move or choose based on Q-values
        for col in range(COLUMNS):
            temp_game = Connect4()
            temp_game.board = game.board.copy()
            temp_game.current_player = -game.current_player  # Switch to the opponent's perspective
            if temp_game.is_valid_location(col):
                temp_game.make_move(col)
                if temp_game.check_winner():  # If this move blocks the opponent's win
                    ai_move = col            # Choose this move
                    break
        else:  # If no blocking move is needed, fallback to MCTS
            root = Node(game)  # Create the root node with the current game state
            ai_move = mcts_search(root, n_iter=n_iter, model=model, device=device)

        # Perform the move in the game
        if not game.is_valid_location(ai_move):
            raise ValueError(f"Invalid move selected by AI: {ai_move}")
        game.make_move(ai_move)               # Update the game state with the selected move
        # print(f"Game {game_num}, Move {move_count}: AI chooses column {ai_move}")  # Debugging
        new_state = game.board.copy()         # Get the updated board state

        # Assign reward and done flag
        if game.check_winner():
            reward = 1 if game.current_player == 1 else -1  # Reward based on winner
            done = True                                              # Game is finished
        elif np.all(game.board != EMPTY):  # Check for draw
            reward = 0
            done = True
        else:
            reward = 0  # Neutral reward for non-terminal states

        # Append to replay buffer for both players
        replay_buffer.append((state.flatten(), ai_move, reward, new_state.flatten(), done))  # AI's experience
        replay_buffer.append((-state.flatten(), ai_move, -reward, -new_state.flatten(), done))  # Opponent's experience

        # Ensure replay buffer doesn't exceed maximum size
        if len(replay_buffer) > MAX_REPLAY_BUFFER_SIZE:
            replay_buffer.pop(0)

        # Switch player
        game.switch_player()  # Change the current player

        # Update current state
        state = new_state   # Update the current state
        move_count += 1     # Increment the move count

def dynamic_n_iter(game_num, total_games, min_iter=1000, max_iter=10000):   # Dynamic number of MCTS iterations           ## min_iter = 1000 for HPC
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
    progress = game_num / total_games  # Calculate progress as a fraction of total games
    n_iter = int(min_iter + (max_iter - min_iter) * (progress ** 2))  # Linearly interpolate iterations
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

# Main Function
def main():
    model = Connect4Net().to(device)  # Initialize the Connect 4 model and move it to the device
    if torch.cuda.device_count() > 1:
        model = DP(model)  # Enable multi-GPU training on a single node
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)  # Initialize optimizer with initial learning rate
    scheduler = get_scheduler(optimizer)  # Initialize the scheduler
    replay_buffer = []  # Initialize an empty replay buffer
    total_games = 50000  # Total number of games to be played
    checkpoint_path = "checkpoint.pth"  # Path to save the checkpoint
    batch_size = 256  # Initial batch size for training
    gamma = 0.995  # Discount factor for future rewards
    loss_history = []  # Store the loss history for dynamic threshold adjustment
    total_steps = 0  # Initialize step counter as a local variable

    # Load from checkpoint if available
    try:
        replay_buffer, start_game = load_checkpoint(checkpoint_path, model, optimizer)
    except FileNotFoundError:
        replay_buffer, start_game = [], 0

    # Self-play and training loop starting from last checkpoint
    for game_num in range(start_game, total_games):
        # Call self_play_game() to simulate a game
        self_play_game(model, replay_buffer, game_num, dynamic_n_iter(game_num, total_games))

        # Train the model
        try:
            loss, loss_history = train_dqn(
                model, optimizer, scheduler, replay_buffer, batch_size=batch_size, gamma=gamma, step=total_steps, loss_history=loss_history
            )
        except RuntimeError as e:
            if "out of memory" in str(e):  # Handle GPU out-of-memory errors
                batch_size //= 2  # Halve the batch size and retry
                # print(f"Out of memory. Reducing batch size to {batch_size} and retrying.")
                loss, loss_history = train_dqn(
                    model, optimizer, scheduler, replay_buffer, batch_size=batch_size, gamma=gamma, step=total_steps, loss_history=loss_history
                )
            else:
                raise e  # Re-raise other exceptions

        total_steps += 1  # Increment the step counter

        # Periodic updates and checkpoint saving
        if (game_num + 1) % 100 == 0:  # Save checkpoint every 100 games
            save_checkpoint(model, optimizer, replay_buffer, game_num + 1, checkpoint_path)

        # Save final checkpoint at the end of training
        if game_num == total_games - 1:
            save_checkpoint(model, optimizer, replay_buffer, game_num + 1, checkpoint_path)

if __name__ == "__main__":
    main()  # Run the main function
