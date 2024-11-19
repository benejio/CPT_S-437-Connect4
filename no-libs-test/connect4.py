

from game import Game




'''
TODO: Make win conditions end game.
TODO: Make tie cause weights to trend toward 0.5
TODO: Make Nodes only recognize columns where leagal move is avaliable
TODO: Make pure random ai 
TODO: MAke graph of amount of turns each game takes duing training
'''

def run_cycle(game, times):
    game.reset_wins()
    for _ in range(times):

        while(game.make_move(game.ai_move(type="best"))):
            pass

        game.reset_game()
    
    game.display_wins()
    


def main():
    # Create a game board
    game = Game(6, 5)
    
    # Display the initial board
    print("Initial board:")
    game.display()
    
    # Simulate some moves

    while(game.is_weighted() == False):
        run_cycle(game, 50000)



    while(game.make_move(game.ai_move(type="best"), debug=True)):
        game.make_move(game.get_player_move(), debug=True)
    print("End board:")
    game.display()
    game.reset_game()

if __name__ == "__main__":
    main()
