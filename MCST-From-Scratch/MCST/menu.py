from node import Node
from board import Board
from model import C4Model
import save_system
import os
import random

class Menu:

    def __init__(self):
        '''
            Inits the menu by starting the main loop
        '''
        self.model = None
        self.running = True
        self.main_loop()


    def main_loop(self):
        '''
            Main loop of menu
        '''
        while(self.running):
            if self.model is None:
                self.choose_model_menu()
            else:
                self.main_menu()

    def create_model_menu(self):
        '''
            Simple menu to create a new model
        '''
        print()
        print("---- Create the Model ----")
        print("Enter grid dimensions")
        rows = int(input("# rows: "))
        cols = int(input("# cols: "))
        self.model = C4Model(rows,cols)

    def load_model_menu(self):
        '''
            Menu for loading the model.
            displays models in the model folder for selection by user
        '''
        print("\n---- Select Model from File ----")
        
        model_folder = "models"
        if not os.path.exists(model_folder):
            print(f"Folder '{model_folder}' does not exist. ")
            return None
        
        files = [f for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f)) and f.endswith('.zip')]

        if not files:
            print(f"No files found in the '{model_folder}' folder.")
            return None
        
        print("Available models:")
        for idx, file in enumerate(files):
            print(f"{idx + 1}. {file}")
        
        while True:
            try:
                choice = int(input("Enter the number of the model you want to load: "))
                if 1 <= choice <= len(files):
                    selected_file = files[choice - 1]
                    break
                else:
                    print(f"Please select a number between 1 and {len(files)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Load the model
        model_path = os.path.join(model_folder, selected_file)
        try:
            saved_node_list, saved_model_params = save_system.load(model_path)
            self.model = C4Model.from_model_attributes(saved_node_list, saved_model_params)
            print(f"Model '{selected_file}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model '{selected_file}': {e}")
            return None

    def choose_model_menu(self):
        print()
        print("---- Choose a Model ----")
        print("Current Model: ", self.model)
        print("1. Create Model")
        print("2. Load Model")
        choice = int(input("Enter choice: "))
        if choice == 1:
            self.create_model_menu()
        elif choice == 2:
            self.load_model_menu()
        else:
            print("invalid input")
        
    def save_model_menu(self):
        '''
            Saves the model to an file based on user input
        '''
        file_name = input("Enter the file name to save to (e.g., 'c4_model_6x7_01.zip'): ").strip()

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        file_path = os.path.join(model_dir, file_name)
        save_system.save_zip(self.model, file_path)

    def play(self, function1, function2):
        '''
            Plays a game of connect 4 where function1 controls player 1 and function2 controls player 2
            Avaliable functions are:
                player_move()
                best_ai()
                random_ai()
                minimax_ai(playerToken)
        '''
        last_move = 1
        while(True):
            last_move = function1()
            self.model.push_move(1, last_move)
            self.model.print()
            if (self.model.tie_detected() or self.model.win_detected(last_move)):
                    break
            self.model.train(self.model.explore_ai, 500,100,2)
            last_move = function2()
            self.model.push_move(2, last_move)
            self.model.print()
            if (self.model.tie_detected() or self.model.win_detected(last_move)):
                    break
            self.model.train(self.model.explore_ai, 500,100,1)
        while(self.model.pop_move()):
            pass

    def main_menu(self):
        print()
        print("---- Menu ----")
        print("Current Model: ", self.model)
        print("1. Unload Model")
        print("2. Save Model")
        print("3. Train Model")
        print("4. Play Human vs AI")
        print("5. Play AI vs AI")
        print("6. Play AI vs Random")
        print("7. Exit")
        choice = int(input("Enter choice: "))
        if choice == 1:
            self.model = None
        elif choice == 2:
            self.save_model_menu()
        elif choice == 3:
            print("Training in progress... Please wait....")
            self.model.train(self.model.explore_ai, 10000, 100, 1)
        elif choice == 4:
            random_number = random.randint(0, 1)
            if random_number == 0:
                print("AI Player Goes First")
                self.play(self.model.best_ai, self.model.player_move)
            else:
                print("Human Player Goes First")
                self.play(self.model.player_move, self.model.best_ai)
        elif choice == 5:
            self.play(self.model.best_ai, self.model.best_ai)
        elif choice == 6:
            random_number = random.randint(0, 1)
            if random_number == 0:
                print("Smart AI Goes First")
                self.play(self.model.best_ai, self.model.random_ai)
            else:
                print("Random AI Goes First")
                self.play(self.model.random_ai, self.model.best_ai)
        elif choice == 7:
            self.running = False
        else:
            print("invalid input")
        



if __name__ == "__main__":
    print()
    print("Welcome to Connect 4 AI Using MCTS")
    Menu()

    