import copy

from games.hnefatafl_stuff.hnefatafl import Hnefatafl
from games.hnefatafl_stuff.position import Position
from self_play import SelfPlay


def test_deduce_latest_move():
    default_board = copy.deepcopy(Hnefatafl.DEFAULT_BOARD)
    updated_board = copy.deepcopy(default_board)
    old_pos = Position(x=3, y=6)
    new_pos = old_pos.left()
    new_pos.set_square(
        board=updated_board, piece=old_pos.get_square(board=updated_board)
    )
    old_pos.set_square(board=updated_board, piece=None)
    move = SelfPlay.deduce_latest_move(
        previous_board=default_board, current_board=updated_board
    )
    assert move[0] == old_pos and move[1] == new_pos
