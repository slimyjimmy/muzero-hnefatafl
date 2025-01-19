import copy
from games.hnefatafl_stuff.hnefatafl import Hnefatafl
from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.position import Position
from games.hnefatafl_stuff.types import default_board


def test_eq():
    pos1 = Position(1, 2)
    pos1b = Position(1, 2)
    pos2 = Position(2, 2)
    assert pos1 == pos1b
    assert not pos1 == pos2


def test_up():
    test = Position(y=1, x=1)
    up = test.up()
    assert up.y == 2 and up.x == 1


def test_down():
    test = Position(y=1, x=1)
    down = test.down()
    assert down.y == 0 and down.x == 1


def test_right():
    test = Position(y=1, x=1)
    right = test.right()
    assert right.y == 1 and right.x == 2


def test_left():
    test = Position(y=1, x=1)
    left = test.left()
    assert left.y == 1 and left.x == 0


def test_get_square():
    board = copy.deepcopy(default_board)
    assert Hnefatafl.MIDDLE.get_square(board=board) == PieceType.KING


def test_set_square():
    board = copy.deepcopy(default_board)
    Hnefatafl.MIDDLE.set_square(board=board, piece=PieceType.ATTACKER)
    assert Hnefatafl.MIDDLE.get_square(board=board) == PieceType.ATTACKER


def test_is_within_board():
    # is in
    assert Hnefatafl.MIDDLE.is_within_board()

    # is over
    assert not Position(y=7, x=1).is_within_board()
    assert not Position(y=8, x=1).is_within_board()

    # is under
    assert not Position(y=-1, x=1).is_within_board()
    assert not Position(y=-2, x=1).is_within_board()

    # is too left
    assert not Position(y=2, x=-1).is_within_board()
    assert not Position(y=2, x=-2).is_within_board()

    # is too right
    assert not Position(y=2, x=7).is_within_board()
    assert not Position(y=2, x=8).is_within_board()


def test_is_open_to_piece():
    # not in field
    assert not Position(7, 7).is_open_to_piece(piece=PieceType.ATTACKER)

    # corner
    assert not Position(0, 0).is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Position(6, 0).is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Position(0, 6).is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Position(6, 6).is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Position(0, 0).is_open_to_piece(piece=PieceType.DEFENDER)
    assert not Position(6, 0).is_open_to_piece(piece=PieceType.DEFENDER)
    assert not Position(0, 6).is_open_to_piece(piece=PieceType.DEFENDER)
    assert not Position(6, 6).is_open_to_piece(piece=PieceType.DEFENDER)
    assert Position(0, 0).is_open_to_piece(piece=PieceType.KING)
    assert Position(6, 0).is_open_to_piece(piece=PieceType.KING)
    assert Position(0, 6).is_open_to_piece(piece=PieceType.KING)
    assert Position(6, 6).is_open_to_piece(piece=PieceType.KING)

    # throne
    assert not Hnefatafl.MIDDLE.is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Hnefatafl.MIDDLE.is_open_to_piece(piece=PieceType.DEFENDER)
    assert Hnefatafl.MIDDLE.is_open_to_piece(piece=PieceType.KING)
