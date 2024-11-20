

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

        while(game.make_move(game.ai_move(type="random"))):
            pass

        game.reset_game()
    
    game.display_wins()
    

def play_vs_ai(game):
    while(game.make_move(game.ai_move(type="best"), debug=True)):
        game.make_move(game.get_player_move(), debug=True)
    print("End board:")
    game.display()
    game.reset_game()

def display_game(game):
    while(game.make_move(game.ai_move(type="best"), debug=True)):
        game.display_weights()
        game.display()
        print("Press any button: ")
        input()
    print("End board:")
    game.display()
    game.reset_game()

def main():
    # Create a game board
    game = Game(4, 4)
    
    # Display the initial board
    print("Initial board:")
    game.display()
    
    # Simulate some moves

    #while(game.is_weighted() == False):
    run_cycle(game, 50000)
    run_cycle(game, 50000)
    run_cycle(game, 50000)
    run_cycle(game, 50000)
    run_cycle(game, 5000)

    print("Node Count: ", game.get_node_count())
    # 4x4 Game Node Count =          71840
    # 4x4 Game with simplification = 19964 ratio 3.59847725907:1

    # 4x5 Game Node Count =          862716
    # 4x5 Game with simplification = 303086

    display_game(game)

    

if __name__ == "__main__":
    main()
