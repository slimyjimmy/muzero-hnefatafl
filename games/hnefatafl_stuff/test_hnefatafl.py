from typing import List
import copy

from games.hnefatafl_stuff.game_result import GameResult
from games.hnefatafl_stuff.hnefatafl import Hnefatafl
from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.player_role import PlayerRole
from games.hnefatafl_stuff.position import Position
from games.hnefatafl_stuff.types import Board

empty_board: Board = [
    [None] * 7,
    [None] * 7,
    [None] * 7,
    [None] * 7,
    [None] * 7,
    [None] * 7,
    [None] * 7,
]

default_board: Board = [
    [
        None,
        None,
        None,
        PieceType.ATTACKER,
        None,
        None,
        None,
    ],
    [
        None,
        None,
        None,
        PieceType.ATTACKER,
        None,
        None,
        None,
    ],
    [
        None,
        None,
        None,
        PieceType.DEFENDER,
        None,
        None,
        None,
    ],
    [
        PieceType.ATTACKER,
        PieceType.ATTACKER,
        PieceType.DEFENDER,
        PieceType.KING,
        PieceType.DEFENDER,
        PieceType.ATTACKER,
        PieceType.ATTACKER,
    ],
    [
        None,
        None,
        None,
        PieceType.DEFENDER,
        None,
        None,
        None,
    ],
    [
        None,
        None,
        None,
        PieceType.ATTACKER,
        None,
        None,
        None,
    ],
    [
        None,
        None,
        None,
        PieceType.ATTACKER,
        None,
        None,
        None,
    ],
]


def test_get_attackers():
    hnefatafl = Hnefatafl()

    attackers = hnefatafl.get_attackers(board=copy.deepcopy(default_board))
    expected_attackers: List[Position] = [
        Position(y=0, x=3),
        Position(y=1, x=3),
        Position(y=3, x=0),
        Position(y=3, x=1),
        Position(y=3, x=5),
        Position(y=3, x=6),
        Position(y=5, x=3),
        Position(y=6, x=3),
    ]
    assert len(attackers) == len(expected_attackers)
    for attacker in attackers:
        assert attacker in expected_attackers


def test_get_defenders():
    hnefatafl = Hnefatafl()

    defenders = hnefatafl.get_defenders(board=copy.deepcopy(default_board))
    expected_defenders: List[Position] = [
        Position(y=2, x=3),
        Position(y=3, x=2),
        Position(y=3, x=4),
        Position(y=4, x=3),
    ]
    assert len(defenders) == len(expected_defenders)
    for defender in defenders:
        assert defender in expected_defenders


def test_get_possible_dests_from_pos():
    hnefatafl = Hnefatafl()

    board_1: Board = [
        [None, PieceType.ATTACKER, None, None, None, None, None],
        [None] * 7,
        [None] * 7,
        [None] * 7,
        [None] * 7,
        [None] * 7,
        [None] * 7,
    ]
    res = hnefatafl.get_possible_dests_from_pos(
        board=board_1, start_pos=Position(y=0, x=1)
    )
    assert len(res) == (7 - 3) + (7 - 1)


def test_get_possible_moves():
    # TODO
    pass


def test_piece_belongs_to_player():
    hnefatafl = Hnefatafl()

    # Attacker, Attacker
    res = hnefatafl.piece_belongs_to_player(
        piece=PieceType.ATTACKER, player=PlayerRole.ATTACKER
    )
    assert res

    # Attacker, Defender
    res = hnefatafl.piece_belongs_to_player(
        piece=PieceType.ATTACKER, player=PlayerRole.DEFENDER
    )
    assert not res

    # None, Attacker
    res = hnefatafl.piece_belongs_to_player(piece=None, player=PlayerRole.ATTACKER)
    assert not res

    # None, Defender
    res = hnefatafl.piece_belongs_to_player(piece=None, player=PlayerRole.DEFENDER)
    assert not res

    # King, Attacker
    res = hnefatafl.piece_belongs_to_player(
        piece=PieceType.KING, player=PlayerRole.ATTACKER
    )
    assert not res

    # King, Defender
    res = hnefatafl.piece_belongs_to_player(
        piece=PieceType.KING, player=PlayerRole.DEFENDER
    )
    assert res


def test_game_over():
    hnefatafl = Hnefatafl()

    board = copy.deepcopy(default_board)

    # king reached corner square -> Defenders win
    corner = Position(0, 0)
    corner.set_square(board=board, piece=PieceType.KING)
    Position(3, 3).set_square(board=board, piece=None)
    res = hnefatafl.game_over(
        king_pos=corner,
        king_captured=False,
        board=board,
        player=PlayerRole.DEFENDER,
        attackers=[],
    )
    assert res[0] == GameResult.WIN and res[1] == PlayerRole.DEFENDER

    # king was captured -> Attackers win
    corner.set_square(board=board, piece=None)
    res = hnefatafl.game_over(
        king_pos=None,
        attackers=[Position(1, 1)],
        board=board,
        king_captured=True,
        player=PlayerRole.ATTACKER,
    )
    assert res[0] == GameResult.WIN and res[1] == PlayerRole.ATTACKER

    # No possible moves
    # TODO

    # No attackers left
    # TODO


def test_is_opponent():
    hnefatafl = Hnefatafl()

    # defender vs defender
    res = hnefatafl.is_opponent(
        piece_type=PieceType.DEFENDER, of_player=PlayerRole.DEFENDER
    )
    assert not res

    # defender vs attacker
    res = hnefatafl.is_opponent(
        piece_type=PieceType.ATTACKER, of_player=PlayerRole.DEFENDER
    )
    assert res

    # defender vs king
    res = hnefatafl.is_opponent(
        piece_type=PieceType.KING, of_player=PlayerRole.DEFENDER
    )
    assert not res

    # defender vs None
    res = hnefatafl.is_opponent(
        piece_type=None,
        of_player=PlayerRole.DEFENDER,
    )
    assert not res

    # attacker vs attacker
    res = hnefatafl.is_opponent(
        piece_type=PieceType.ATTACKER, of_player=PlayerRole.ATTACKER
    )
    assert not res

    # attacker vs defender
    res = hnefatafl.is_opponent(
        piece_type=PieceType.DEFENDER, of_player=PlayerRole.ATTACKER
    )
    assert res

    # attacker vs king
    res = hnefatafl.is_opponent(
        piece_type=PieceType.KING, of_player=PlayerRole.ATTACKER
    )
    assert res

    # attacker vs None
    res = hnefatafl.is_opponent(piece_type=None, of_player=PlayerRole.ATTACKER)
    assert not res


def test_get_rendering_string():
    hnefatafl = Hnefatafl()

    board = copy.deepcopy(default_board)
    res = hnefatafl.get_rendering_string(board=board)
    print(f"at (3,3) is: {Position(3,3).get_square(board=board)}")
    assert (
        res
        == "  A   B   C   D   E   F   G\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 7\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 6\n+---+---+---+---+---+---+---+\n|   |   |   | 🛡️ |   |   |   | 5\n+---+---+---+---+---+---+---+\n| 🗡️ | 🗡️ | 🛡️ | K | 🛡️ | 🗡️ | 🗡️ | 4\n+---+---+---+---+---+---+---+\n|   |   |   | 🛡️ |   |   |   | 3\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 2\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 1\n+---+---+---+---+---+---+---+\n"
    )


def test_piece_captured():
    # TODO
    pass


def test_piece_captures_opponent():
    # TODO
    pass


def test_king_is_captured():
    hnefatafl = Hnefatafl()

    # king is on throne and surrounded by attackers
    middle_captured = copy.deepcopy(default_board)
    Position(y=2, x=3).set_square(board=middle_captured, piece=PieceType.ATTACKER)
    Position(y=3, x=2).set_square(board=middle_captured, piece=PieceType.ATTACKER)
    Position(y=3, x=4).set_square(board=middle_captured, piece=PieceType.ATTACKER)
    Position(y=4, x=3).set_square(board=middle_captured, piece=PieceType.ATTACKER)
    assert hnefatafl.king_is_captured(Position(3, 3), middle_captured)

    # king is on throne, but not fully surrounded by attackers
    Position(y=4, x=3).set_square(board=middle_captured, piece=None)
    assert not hnefatafl.king_is_captured(Position(3, 3), middle_captured)

    # king is next to throne and surrounded by attackers on other 3 sides
    next_to_throne_captured = copy.deepcopy(empty_board)
    king_pos = Position(y=4, x=3)
    king_pos.set_square(board=next_to_throne_captured, piece=PieceType.KING)
    king_pos.up().set_square(board=next_to_throne_captured, piece=PieceType.ATTACKER)
    king_pos.left().set_square(board=next_to_throne_captured, piece=PieceType.ATTACKER)
    king_pos.right().set_square(board=next_to_throne_captured, piece=PieceType.ATTACKER)
    assert hnefatafl.king_is_captured(king_pos=king_pos, board=next_to_throne_captured)

    # king is on "random" square (not restricted, not throne, not next to throne) and captured
    random_captured = copy.deepcopy(empty_board)
    king_pos = Position(2, 1)
    king_pos.set_square(board=random_captured, piece=PieceType.KING)
    king_pos.up().set_square(board=random_captured, piece=PieceType.ATTACKER)
    king_pos.down().set_square(board=random_captured, piece=PieceType.ATTACKER)
    king_pos.left().set_square(board=random_captured, piece=PieceType.ATTACKER)
    king_pos.right().set_square(board=random_captured, piece=PieceType.ATTACKER)
    assert hnefatafl.king_is_captured(king_pos=king_pos, board=random_captured)

    king_pos.right().set_square(board=random_captured, piece=None)
    assert not hnefatafl.king_is_captured(king_pos=king_pos, board=random_captured)
