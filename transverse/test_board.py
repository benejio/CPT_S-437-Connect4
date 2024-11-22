import pytest
from board import Board

def test_out_of_range_col():
    game = Board()
    col = 10
    token = 1
    result = game.drop_token(token, col)
    assert result == False

def test_in_range_col():
    game = Board()
    col = 3
    token = 1
    result = game.drop_token(token, col)
    assert result == True

