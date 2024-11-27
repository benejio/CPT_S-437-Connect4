# Import necessary libraries and modules
import torch                                            # PyTorch for model handling
import random                                           # For random move selection
import numpy as np                                      # For numerical operations
from Connect4 import Connect4, ROWS, COLUMNS, EMPTY     # Import Connect4 game class and constants
from network import Connect4Net                         # Import the neural network for Connect4
from train import load_checkpoint                       # Import the function to load a trained model checkpoint

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play_against_model(model_path="checkpoint.pth"):
    # Load the neural network model
    model = Connect4Net().to(device)                            # Initialize the model and move it to the selected device
    model.eval()                                                # Set the model to evaluation mode (disables training-specific layers like dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer is needed for loading checkpoint

    # Load the trained model weights and optimizer state from the checkpoint
    _, _ = load_checkpoint(model_path, model, optimizer)

    # Initialize the Connect4 game
    game = Connect4()   # Create a new Connect4 game instance
    game.reset()        # Reset the game board to its initial state
    done = False        # Flag to indicate whether the game is over

    # Print instructions for the player
    print("Welcome to Connect 4! You are Player 1. Enter a column number (0-6) to make a move.")

    while not done:         # Loop until the game is finished
        game.print_board()      # Print the current game board

        # Player move
        player_move = int(input("Your move (0-6): "))   # Get player's move as input
        if not game.make_move(player_move):             # Validate and make the move
            print("Invalid move. Try again.")           # Notify if the move is invalid
            continue                                    # Retry if the move was invalid

        if game.check_winner():                         # Check if the player has won
            game.print_board()                          # Print the final board state
            print("Congratulations! You win!")          # Notify the player of their victory
            break                                       # End the game loop

        game.switch_player()  # Switch to the AI player

        # AI move
        # Prepare the current game board as input to the AI model
        state_tensor = torch.tensor(game.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # print(f"Shape of state_tensor: {state_tensor.shape}")  # Debugging: Check input tensor shape

        # Predict Q-values for each column using the AI model
        with torch.no_grad():   # Disable gradient computation for efficiency
            q_values = model(state_tensor)
        # print(f"Q-values for AI: {q_values}")  # Debugging: Inspect the AI's predicted Q-values

        # Choose the AI move based on Q-values or block the opponent's winning move
        for col in range(COLUMNS):                              # Iterate through all possible columns
            temp_game = Connect4()                              # Create a temporary game instance
            temp_game.board = game.board.copy()                 # Copy the current board state
            temp_game.current_player = -game.current_player     # Set the opponent's perspective
            if temp_game.is_valid_location(col):                # Check if the column is a valid move
                temp_game.make_move(col)                        # Make the move in the temporary game
                if temp_game.check_winner():                    # Check if this move blocks the opponent's win
                    ai_move = col                               # Choose this move to block the win
                    break
        else:
            # Default to Q-value-based decision
            if torch.allclose(q_values, q_values[0]):  # If all Q-values are very similar, choose randomly
                ai_move = random.choice([col for col in range(COLUMNS) if game.is_valid_location(col)])
            else:
                ai_move = torch.argmax(q_values).item()  # Choose the column with the highest Q-value

        print(f"AI selects column {ai_move}")   # Notify the player of the AI's move
        game.make_move(ai_move)                 # Make the AI's move on the actual game board

        if game.check_winner():                         # Check if the AI has won
            game.print_board()                              # Print the final board state
            print("AI wins! Better luck next time.")        # Notify the player of the AI's victory
            done = True                                     # Mark the game as finished
        elif np.all(game.board != EMPTY):               # Check if the board is full (a draw)
            game.print_board()                              # Print the final board state
            print("It's a draw!")                           # Notify the player of the draw
            done = True                                     # Mark the game as finished

        game.switch_player()  # Switch back to the human player after the AI's move

# Entry point for the script
if __name__ == "__main__":
    play_against_model()  # Call the function to play against the AI model
