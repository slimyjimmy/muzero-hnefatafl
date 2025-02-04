import copy
from games.hnefatafl_stuff.direction import Direction
from games.hnefatafl_stuff.hnefatafl import Hnefatafl
from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.position import Position
from games.hnefatafl_stuff.types import default_board


def test_hash():
    p1 = Position(x=1, y=2)
    p2 = Position(x=1, y=2)
    assert hash(p1) == hash(p2)
    p1 = Position(x=1, y=2)
    p2 = Position(x=2, y=3)
    assert hash(p1) != hash(p2)


def test_eq():
    pos1 = Position(x=1, y=2)
    pos1b = Position(x=1, y=2)
    pos2 = Position(x=2, y=2)
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
    invalid_pos = Position(x=Position.INVALID, y=1)
    assert invalid_pos.get_square(board=board) == None


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


def test_to_list():
    pos = Position(x=1, y=2)
    list = pos.to_list()
    assert len(list) == 2 and list[0] == 1 and list[1] == 2


def test_to_string():
    pos = Position(x=1, y=2)
    assert pos.to_string() == "(B5)"


def test_is_occupied():
    default_board = copy.deepcopy(Hnefatafl.DEFAULT_BOARD)
    assert Position(x=3, y=3).is_occupied(board=default_board)
    assert not Position(x=1, y=6).is_occupied(board=default_board)


def test_is_open_to_piece():
    # no piece given
    assert not Position(x=1, y=2).is_open_to_piece(piece=None)

    # not in field
    assert not Position(x=7, y=7).is_open_to_piece(piece=PieceType.ATTACKER)

    # corner
    assert not Position(x=0, y=0).is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Position(x=6, y=0).is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Position(x=0, y=6).is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Position(x=6, y=6).is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Position(x=0, y=0).is_open_to_piece(piece=PieceType.DEFENDER)
    assert not Position(x=6, y=0).is_open_to_piece(piece=PieceType.DEFENDER)
    assert not Position(x=0, y=6).is_open_to_piece(piece=PieceType.DEFENDER)
    assert not Position(x=6, y=6).is_open_to_piece(piece=PieceType.DEFENDER)
    assert Position(x=0, y=0).is_open_to_piece(piece=PieceType.KING)
    assert Position(x=6, y=0).is_open_to_piece(piece=PieceType.KING)
    assert Position(x=0, y=6).is_open_to_piece(piece=PieceType.KING)
    assert Position(x=6, y=6).is_open_to_piece(piece=PieceType.KING)

    # throne
    assert not Hnefatafl.MIDDLE.is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Hnefatafl.MIDDLE.is_open_to_piece(piece=PieceType.DEFENDER)
    assert Hnefatafl.MIDDLE.is_open_to_piece(piece=PieceType.KING)


def test_get_adjacent_position():
    pos = Position(y=2, x=2)

    assert pos.get_adjacent_position(direction=None) == None

    # down
    assert pos.get_adjacent_position(direction=Direction.DOWN) == pos.down(steps=1)
    assert pos.get_adjacent_position(direction=Direction.DOWN, steps=2) == pos.down(
        steps=2
    )
    # up
    assert pos.get_adjacent_position(direction=Direction.UP) == pos.up(steps=1)
    assert pos.get_adjacent_position(direction=Direction.UP, steps=2) == pos.up(steps=2)
    # left
    assert pos.get_adjacent_position(direction=Direction.LEFT) == pos.left(steps=1)
    assert pos.get_adjacent_position(direction=Direction.LEFT, steps=2) == pos.left(
        steps=2
    )
    # right
    assert pos.get_adjacent_position(direction=Direction.RIGHT) == pos.right(steps=1)
    assert pos.get_adjacent_position(direction=Direction.RIGHT, steps=2) == pos.right(
        steps=2
    )


def test_is_adjacent():
    pos = Position(x=2, y=2)

    assert pos.is_adjacent(None) == None

    # down
    assert pos.is_adjacent(pos.down()) == Direction.DOWN

    # up
    assert pos.is_adjacent(pos.up()) == Direction.UP

    # left
    assert pos.is_adjacent(pos.left()) == Direction.LEFT

    # right
    assert pos.is_adjacent(pos.right()) == Direction.RIGHT
