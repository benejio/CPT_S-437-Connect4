# CPT_S-437-Connect4
Machine learning models to play connect 4. 

This GitHub repo contains two different reinforcment learning models.
1. DeepQ Learning with MCST
2. MCST from scratch

### DeepQ Learning Implementation


### MCST From Scratch Implementation

This algorithm, Monte Carlo Search Tree, iterativly plays the game against itself and records winning moves. Through backpropigation these winning (or losing moves) impact the weights of moves earlier in the game that lead to the moves.

Boardstates are represented as Nodes. Each Node contains weights for each possible move leading from it to another Node. The weight is calculated based on backproigation of the likelyhood for the next player to make a move.

#### MCST Optimizations

MCST requires a lot of calculations for larger board sizes. This is because of the exponentially increasing complexity of the problem as more spaces are introduced.
To alleviate this calculation load there are a couple of tricks we implemented.

1. Nodes do not care which moves lead to its boardstate. The means that the same boardstate reached in 2 different ways is represented by the same node.

2. Boards are evaluated based only on relevant tokens. This amakes it so boards that have functionally equvalent board states all point to the same node. This reduces the amount of total nodes required (especially on larger boards)

3. While training, instead of playing a full game, we evaluate game states in batches where we pop back a single move to test a different path.

4. In order to speed up training, each node has an array that indecates if it has been explored or not. This insures we are not recalculating the same paths.

5. Lastly, instead of only training before the game is played, the model will train after every move made. This insures that the model will at least have some idea of what moves are best in any given state. Also, because complexity reduced massively with each move, we can learn which moves are best in each given state easier.


### Required Libraries

```terminal
# For DeepQ Learning Model
torch
tkinter
concurrent
numpy
random
os
```

```terminal
# For MCST From Scratch Model
numpy
random
math
os
json
gzip
```

