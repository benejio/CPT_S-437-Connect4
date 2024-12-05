class Position:
    WIDTH = 7  # width of the board
    HEIGHT = 6  # height of the board

    def __init__(self):
        self.current_position = 0
        self.mask = 0
        self.moves = 0

    def can_play(self, col):
        """Indicates whether a column is playable."""
        return (self.mask & self.top_mask(col)) == 0

    def play(self, col):
        """Plays a playable column."""
        self.current_position ^= self.mask
        self.mask |= self.mask + self.bottom_mask(col)
        self.moves += 1

    def play_sequence(self, seq):
        """Plays a sequence of successive played columns."""
        for i in range(len(seq)):
            col = int(seq[i]) - 1
            if col < 0 or col >= Position.WIDTH or not self.can_play(col) or self.is_winning_move(col):
                return i  # invalid move
            self.play(col)
        return len(seq)

    def is_winning_move(self, col):
        """Indicates whether the current player wins by playing a given column."""
        pos = self.current_position
        pos |= (self.mask + self.bottom_mask(col)) & self.column_mask(col)
        return self.alignment(pos)

    def nb_moves(self):
        """Returns the number of moves played."""
        return self.moves

    def key(self):
        """Returns a compact representation of a position."""
        return self.current_position + self.mask

    def alignment(self, pos):
        """Test an alignment for the current player (identified by 1 in the bitboard pos)."""
        # horizontal
        m = pos & (pos >> (Position.HEIGHT + 1))
        if m & (m >> (2 * (Position.HEIGHT + 1))):
            return True

        # diagonal 1
        m = pos & (pos >> Position.HEIGHT)
        if m & (m >> (2 * Position.HEIGHT)):
            return True

        # diagonal 2
        m = pos & (pos >> (Position.HEIGHT + 2))
        if m & (m >> (2 * (Position.HEIGHT + 2))):
            return True

        # vertical
        m = pos & (pos >> 1)
        if m & (m >> 2):
            return True

        return False

    def top_mask(self, col):
        """Returns a bitmask containing a single 1 corresponding to the top cell of a given column."""
        return 1 << (Position.HEIGHT - 1 + col * (Position.HEIGHT + 1))

    def bottom_mask(self, col):
        """Returns a bitmask containing a single 1 corresponding to the bottom cell of a given column."""
        return 1 << col * (Position.HEIGHT + 1)

    def column_mask(self, col):
        """Returns a bitmask with 1 on all the cells of a given column."""
        return (1 << Position.HEIGHT) - 1 << col * (Position.HEIGHT + 1)

    def copy(self):
        """Creates a copy of the current Position object."""
        new_position = Position()
        new_position.current_position = self.current_position
        new_position.mask = self.mask
        new_position.moves = self.moves
        return new_position
