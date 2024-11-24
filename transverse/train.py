

from node import Node
from board import Board
from model import C4Model




'''
TODO: Make it so the program learns based of a current state.
      This can be used to reduce complexity greatly and assist with later moves.
      For example after each player move simulate 10000 possible games and learn from them.
'''

model = C4Model(6, 7)


model.print()
model.board.drop_token("1",1)
model.board.drop_token("2",1)
model.board.drop_token("1",2)
node = model.get_node()
print(node.id)
model.board.drop_token("2",3)

model.print()
print(model.move_history[0].id)
model.board.load_node(model.move_history[0].id)
model.print()

print(model.random_ai())
model.board.load_node(node.id)
model.print()