

from node import Node
from board import Board
from model import C4Model




'''
TODO: Make it so the program learns based of a current state.
      This can be used to reduce complexity greatly and assist with later moves.
      For example after each player move simulate 10000 possible games and learn from them.

TODO: Add % explored after each move
'''

model = C4Model(5, 5)




model.print()
model.train(model.explore_ai, 10000, 100, 1)

last_move = 1
while(True):
      last_move = model.player_move()
      model.push_move(1, last_move)
      model.print()
      if (model.tie_detected() or model.win_detected(last_move)):
            break
      model.train(model.explore_ai, 5000,50,2)
      last_move = model.best_ai()
      model.push_move(2, last_move)
      model.print()
      if (model.tie_detected() or model.win_detected(last_move)):
            break

