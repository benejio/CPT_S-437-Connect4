# play.py

import torch
import random
import numpy as np
from Connect4 import Connect4, ROWS, COLUMNS, EMPTY
from network import Connect4Net
from train import load_checkpoint  # Import the load_checkpoint function from train.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play_against_model(model_path="checkpoint.pth"):
    # Load the model
    model = Connect4Net().to(device)
    model.eval()  # Set model to evaluation mode
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer is needed for loading checkpoint

    # Load checkpoint
    _, _ = load_checkpoint(model_path, model, optimizer)

    # Initialize game
    game = Connect4()
    game.reset()
    done = False

    print("Welcome to Connect 4! You are Player 1. Enter a column number (0-6) to make a move.")
    while not done:
        game.print_board()
        
        # Player move
        player_move = int(input("Your move (0-6): "))
        if not game.make_move(player_move):
            print("Invalid move. Try again.")
            continue

        if game.check_winner():
            game.print_board()
            print("Congratulations! You win!")
            break

        game.switch_player()  # Switch to AI player

        # AI move
        state_tensor = torch.tensor(game.board.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        print(f"Q-values for AI: {q_values}")  # Debugging line to inspect Q-values
        
        # Choose AI move based on Q-values or random if values are similar
        if torch.allclose(q_values, q_values[0]):  # If Q-values are very similar, choose randomly
            ai_move = random.choice([col for col in range(COLUMNS) if game.is_valid_location(col)])
        else:
            ai_move = torch.argmax(q_values).item()

        print(f"AI selects column {ai_move}")
        game.make_move(ai_move)
        
        if game.check_winner():
            game.print_board()
            print("AI wins! Better luck next time.")
            done = True
        elif np.all(game.board != EMPTY):
            game.print_board()
            print("It's a draw!")
            done = True

        game.switch_player()  # Switch back to human player after AI move


if __name__ == "__main__":
    play_against_model()
