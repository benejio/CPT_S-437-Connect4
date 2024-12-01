import tkinter as tk                                     # GUI library for creating the Connect 4 interface
from tkinter import messagebox                          # For displaying popup messages in the GUI
import torch                                            # PyTorch for loading and running the AI model
import random                                           # For selecting random moves when necessary
import numpy as np                                      # For numerical operations on the game board
from Connect4 import Connect4, ROWS, COLUMNS, EMPTY     # Import Connect 4 game logic and constants
from network import Connect4Net                         # Import the neural network model for Connect 4
from train import load_checkpoint                       # Import function to load a trained model checkpoint

# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

class Connect4GUI:
    def __init__(self, root, model_path="checkpoint.pth"):
        self.root = root                                # Reference to the root Tkinter window
        self.root.title("Connect 4")                   # Set the title of the window

        # Initialize model
        self.model = Connect4Net().to(device)          # Load the Connect 4 neural network model
        self.model.eval()                              # Set the model to evaluation mode
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # Dummy optimizer for loading checkpoint
        _, _ = load_checkpoint(model_path, self.model, optimizer)        # Load the trained model weights

        # Initialize game
        self.game = Connect4()                         # Create a new Connect 4 game instance
        self.game.reset()                              # Reset the game to its initial state

        # Create board UI
        self.buttons = [                               # Create a list of buttons for dropping pieces into columns
            tk.Button(root, text=f"Drop {col}", command=lambda c=col: self.player_move(c))
            for col in range(COLUMNS)
        ]
        for col, btn in enumerate(self.buttons):       # Place the buttons at the top of the window
            btn.grid(row=0, column=col)

        self.cells = [                                 # Create labels for each cell in the Connect 4 board
            [tk.Label(root, text=" ", width=5, height=2, bg="blue", fg="white", relief="ridge")
             for _ in range(COLUMNS)] for _ in range(ROWS)
        ]
        for r in range(ROWS):                          # Place the labels in a grid below the buttons
            for c in range(COLUMNS):
                self.cells[r][c].grid(row=r+1, column=c)

    def player_move(self, col):
        if not self.game.make_move(col):               # Check if the move is valid; if not, show an error message
            messagebox.showerror("Invalid Move", "Column is full. Try another!")  # Display error
            return                                     # Exit if the move is invalid

        self.update_board()                            # Update the board display after the player's move
        if self.game.check_winner():                  # Check if the player has won
            messagebox.showinfo("You Win!", "Congratulations! You have won the game!")  # Show win message
            self.reset_game()                         # Reset the game
            return
        elif np.all(self.game.board != EMPTY):        # Check for a draw (board is full)
            messagebox.showinfo("Draw", "It's a draw!")  # Show draw message
            self.reset_game()                         # Reset the game
            return

        self.game.switch_player()                     # Switch to the AI player
        self.ai_move()                                # Let the AI make its move

    def ai_move(self):
        """
        Let the AI calculate and perform its move using Q-values from the model or by blocking the opponent.
        """
        state_tensor = torch.tensor(                  # Convert the current board state to a PyTorch tensor
            self.game.board, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():                         # Disable gradient computation for inference
            q_values = self.model(state_tensor)       # Predict Q-values for each column using the AI model

        # Check for a winning move for the AI
        for col in range(COLUMNS):
            temp_game = Connect4()
            temp_game.board = self.game.board.copy()
            temp_game.current_player = self.game.current_player  # AI's perspective
            if temp_game.is_valid_location(col):                # Ensure the column is valid
                temp_game.make_move(col)
                if temp_game.check_winner():                   # Check if this move wins the game for AI
                    ai_move = col                              # AI selects this move
                    print(f"AI chooses winning move: Column {col}")  # Debugging statement
                    break
        else:  # If no winning move is found
            # Block opponent's winning move
            for col in range(COLUMNS):
                temp_game = Connect4()
                temp_game.board = self.game.board.copy()
                temp_game.current_player = -self.game.current_player  # Opponent's perspective
                if temp_game.is_valid_location(col):
                    temp_game.make_move(col)
                    if temp_game.check_winner():  # If this move blocks the opponent's win
                        ai_move = col            # Choose this move to block
                        print(f"AI blocks opponent's winning move: Column {col}")  # Debugging statement
                        break
            else:  # If no blocking is needed
                if torch.allclose(q_values, q_values[0]):  # Check if all Q-values are similar (no strong preference)
                    ai_move = random.choice([col for col in range(COLUMNS) if self.game.is_valid_location(col)])
                    print(f"AI chooses random move: Column {ai_move}")  # Debugging statement
                else:
                    ai_move = torch.argmax(q_values).item()  # Choose the column with the highest Q-value
                    print(f"AI chooses best Q-value move: Column {ai_move}")  # Debugging statement

        # Perform the AI's move
        if not self.game.make_move(ai_move):         # Make the AI's move and ensure it's valid
            print("AI attempted an invalid move.")   # Debugging statement
            return

        # Update the board display after AI's move
        self.update_board()

        # Check if the AI has won
        if self.game.check_winner():
            messagebox.showinfo("AI Wins", "AI wins! Better luck next time.")  # Notify user of AI's win
            self.reset_game()  # Reset the game
        elif np.all(self.game.board != EMPTY):      # Check for a draw (board is full)
            messagebox.showinfo("Draw", "It's a draw!")  # Notify user of a draw
            self.reset_game()  # Reset the game

        # Switch back to the human player
        self.game.switch_player()

    def update_board(self):
        for r in range(ROWS):                       # Iterate through all rows
            for c in range(COLUMNS):                # Iterate through all columns
                cell_value = self.game.board[r][c]  # Get the value of the current cell
                if cell_value == 1:                 # If the cell belongs to Player 1
                    self.cells[r][c]["text"] = "X"  # Display "X"
                    self.cells[r][c]["bg"] = "red"  # Set background to red
                elif cell_value == -1:              # If the cell belongs to Player -1 (AI)
                    self.cells[r][c]["text"] = "O"  # Display "O"
                    self.cells[r][c]["bg"] = "yellow"  # Set background to yellow
                else:                               # If the cell is empty
                    self.cells[r][c]["text"] = " "  # Clear text
                    self.cells[r][c]["bg"] = "blue"  # Set background to blue

    def reset_game(self):
        self.game.reset()                           # Reset the game state
        self.update_board()                         # Clear the board display

# Run the application
if __name__ == "__main__":
    root = tk.Tk()                                  # Create the main Tkinter window
    app = Connect4GUI(root)                         # Create the Connect 4 GUI
    root.mainloop()                                 # Start the Tkinter event loop
