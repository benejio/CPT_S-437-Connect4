# train.py

import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Connect4 import Connect4, EMPTY
from network import Connect4Net
from mcts import mcts_search, Node

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dqn(model, optimizer, replay_buffer, batch_size=16, gamma=0.995, step=0):
    if len(replay_buffer) < batch_size:
        return None  # Not enough experience to train

    # Sample a batch of experiences
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors and move to device
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

    # Compute Q-values for current states
    q_values = model(states).gather(1, actions).squeeze()

    # Compute target Q-values for next states
    next_q_values = model(next_states).max(1)[0]
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Calculate the loss
    loss = F.mse_loss(q_values, target_q_values.detach())
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()

    # Print loss every 100 steps for monitoring
    if step % 100 == 0:
        print(f"Training step {step}: Loss = {loss.item()}")

    return loss.item()

def self_play_game(model, replay_buffer, game_num):
    game = Connect4()
    game.reset()
    state = game.board.copy()
    done = False
    move_count = 0

    while not done:
        root = Node(game)  # Root node with the current game state
        best_move = mcts_search(root, n_iter=1000, model=model, device=device)

        ### print(f"Game {game_num}, Move {move_count}, Player {game.current_player} selects column {best_move}")

        # Perform the move in the game
        game.make_move(best_move)
        new_state = game.board.copy()
        
        # Assign reward and done flag
        reward = 1 if game.check_winner() else 0  # Positive reward if there's a winner
        done = reward == 1 or np.all(game.board != EMPTY)  # Game ends on win or board full
        
        # Append to replay buffer
        replay_buffer.append((state.flatten(), best_move, reward, new_state.flatten(), done))
        
        # Switch player
        game.switch_player()
        
        # Update current state
        state = new_state
        move_count += 1

    ### print(f"Game {game_num} ended. Total moves: {move_count}")

def load_checkpoint(file_path, model, optimizer):
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        replay_buffer = checkpoint.get('replay_buffer', [])
        game_num = checkpoint.get('game_num', 0)
        print(f"Checkpoint loaded from {file_path} (game {game_num})")
        return replay_buffer, game_num
    else:
        print(f"No checkpoint found at {file_path}")
        return [], 0

def save_checkpoint(model, optimizer, replay_buffer, game_num, file_path="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer': replay_buffer,  # To save experience buffer
        'game_num': game_num
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at game {game_num} to {file_path}")

def main():
    model = Connect4Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    replay_buffer = []
    total_games = 20000
    total_steps = 0
    checkpoint_path = "checkpoint.pth"

    # Load from checkpoint if available
    replay_buffer, start_game = load_checkpoint(checkpoint_path, model, optimizer)

    # Self-play and training loop starting from last checkpoint
    for game_num in range(start_game, total_games):
        print(f"Starting game {game_num + 1}/{total_games}")
        self_play_game(model, replay_buffer, game_num + 1)

        # Training step after each game
        loss = train_dqn(model, optimizer, replay_buffer, batch_size=16, step=total_steps)
        
        if loss is not None:
            print(f"Game {game_num + 1}: Training loss after game = {loss:.4f}")

        total_steps += 1

        # Periodic updates and checkpoint saving
        if (game_num + 1) % 100 == 0:
            print(f"Status: {game_num + 1} games played, Replay buffer size: {len(replay_buffer)}")
            save_checkpoint(model, optimizer, replay_buffer, game_num + 1, checkpoint_path)

if __name__ == "__main__":
    main()
