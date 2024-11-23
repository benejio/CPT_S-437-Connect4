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

def test_generate_id():
    game = Board()
    generated_id = game.generate_id()
    generated_id2 = game.generate_id()
    assert generated_id == generated_id2

def test_generate_different_id():
    game = Board()
    generated_id = game.generate_id()
    game.drop_token(1, 3)
    generated_id2 = game.generate_id()
    assert generated_id != generated_id2

