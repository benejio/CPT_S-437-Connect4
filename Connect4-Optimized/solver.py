from position import Position
import time
from typing import Tuple

class Solver:
    def __init__(self):
        self.node_count = 0
        self.column_order = [0] * Position.WIDTH  # Initialize the column order list
        for i in range(Position.WIDTH):
            self.column_order[i] = Position.WIDTH // 2 + (1 - 2 * (i % 2)) * (i + 1) // 2


    def negamax(self, P: Position, alpha: int, beta: int) -> int:
        assert alpha < beta
        self.node_count += 1

        if P.nb_moves() >= Position.WIDTH * Position.HEIGHT:
            return 0

        for x in range(Position.WIDTH):
            if P.can_play(x) and P.is_winning_move(x):
                return (Position.WIDTH * Position.HEIGHT + 1 - P.nb_moves()) // 2

        max_score = (Position.WIDTH * Position.HEIGHT - 1 - P.nb_moves()) // 2
        if beta > max_score:
            beta = max_score
            if alpha >= beta:
                return beta
        
        for x in range(Position.WIDTH):
            if P.can_play(self.column_order[x]):
                P2 = P.copy()
                P2.play(self.column_order[x])
                score = -self.negamax(P2, -beta, -alpha)
                if score >= beta:
                    return score
                if score > alpha:
                    alpha = score

        return alpha

    def solve(self, P: Position, weak: bool = False) -> int:
        self.node_count = 0
        print("Solving")
        if weak:
            return self.negamax(P, -1, 1)
        else:
            return self.negamax(P, -Position.WIDTH * Position.HEIGHT // 2, Position.WIDTH * Position.HEIGHT // 2)


    def get_node_count(self) -> int:
        return self.node_count

def get_time_microsec() -> int:
    return int(time.time() * 1_000_000)

def main():
    solver = Solver()
    weak = False

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '-w':
        print("Weak")
        weak = True

    for l, line in enumerate(sys.stdin, start=1):
        print(l, line)
        line = line.strip()
        P = Position()
        
        if P.play_sequence(line) != len(line):
            print(f"Line {l}: Invalid move {P.nb_moves() + 1} \"{line}\"", file=sys.stderr)
        else:
            start_time = get_time_microsec()
            score = solver.solve(P, weak)
            end_time = get_time_microsec()
            print(f"{line} {score} {solver.get_node_count()} {end_time - start_time}")
        print()


if __name__ == "__main__":
    main()