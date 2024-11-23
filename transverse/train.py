


from node import Node
from board import Board


board = Board()

'''
TODO: Make it so the program learns based of a current state.
      This can be used to reduce complexity greatly and assist with later moves.
      For example after each player move simulate 10000 possible games and learn from them.
'''

board.print()
print()

board.drop_token('x',1)

board.drop_token('o',1)

board.print()