from typing import List
import unittest

from games.hnefatafl_stuff.hnefatafl import Hnefatafl
from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.player_role import PlayerRole
from games.hnefatafl_stuff.position import Position
from games.hnefatafl_stuff.types import Board


def test_get_attackers():
    hnefatafl = Hnefatafl()
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
    attackers = hnefatafl.get_attackers(board=default_board)
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
    defenders = hnefatafl.get_defenders(board=default_board)
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


"""def test_get_possible_moves():
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
    res = hnefatafl.get_possible_moves(board=board_1, player=PlayerRole.ATTACKER)
    assert len(res) == (7 - 3) + (7 - 1)"""


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


"""def test_game_over():
    raise NotImplementedError("This method is not implemented yet.")"""


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

    board_1: Board = [
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
    res = hnefatafl.get_rendering_string(board=board_1)
    assert (
        res
        == "  A   B   C   D   E   F   G\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 7\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 6\n+---+---+---+---+---+---+---+\n|   |   |   | 🛡️ |   |   |   | 5\n+---+---+---+---+---+---+---+\n| 🗡️ | 🗡️ | 🛡️ | K | 🛡️ | 🗡️ | 🗡️ | 4\n+---+---+---+---+---+---+---+\n|   |   |   | 🛡️ |   |   |   | 3\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 2\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 1\n+---+---+---+---+---+---+---+\n"
    )
