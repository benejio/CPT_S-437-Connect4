

from game import Game




'''
TODO: Make win conditions end game.
TODO: Make tie cause weights to trend toward 0.5
TODO: Make Nodes only recognize columns where leagal move is avaliable
TODO: Make pure random ai 
TODO: MAke graph of amount of turns each game takes duing training
'''

def run_cycle(game, times):
    for _ in range(times):

        while(game.make_move(game.ai_move())):
            pass

        game.reset_game()
    
    game.display_wins()
    game.reset_wins()


def main():
    # Create a game board
    game = Game(5, 5)
    
    # Display the initial board
    print("Initial board:")
    game.display()
    
    # Simulate some moves
    
    run_cycle(game, 500000)
    run_cycle(game, 500000)
    run_cycle(game, 500000)

    while(game.make_move(game.ai_move(), debug=True)):
        pass
    print("\End board:")
    game.display()
    game.reset_game()
    print("Node count:", game.get_node_count())



    

if __name__ == "__main__":
    main()
