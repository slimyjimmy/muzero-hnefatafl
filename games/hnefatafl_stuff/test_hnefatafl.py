from typing import List
import copy

from games.hnefatafl_stuff.game_result import GameResult
from games.hnefatafl_stuff.hnefatafl import Hnefatafl
from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.player_role import PlayerRole
from games.hnefatafl_stuff.position import Position
from games.hnefatafl_stuff.types import Board, default_board, empty_board


def test_get_attackers():
    attackers = Hnefatafl.get_attackers(copy.deepcopy(default_board))
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
    defenders = Hnefatafl.get_defenders(board=copy.deepcopy(default_board))
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
    board_1: Board = [
        [None, PieceType.ATTACKER, None, None, None, None, None],
        [None] * 7,
        [None] * 7,
        [None] * 7,
        [None] * 7,
        [None] * 7,
        [None] * 7,
    ]
    res = Hnefatafl.get_possible_dests_from_pos(
        board=board_1,
        start_pos=Position(y=0, x=1),
        player=PlayerRole.ATTACKER,
    )
    assert len(res) == (7 - 3) + (7 - 1)


def test_get_possible_moves():
    # TODO
    pass


def test_piece_belongs_to_player():
    # Attacker, Attacker
    res = Hnefatafl.piece_belongs_to_player(
        piece=PieceType.ATTACKER, player=PlayerRole.ATTACKER
    )
    assert res

    # Attacker, Defender
    res = Hnefatafl.piece_belongs_to_player(
        piece=PieceType.ATTACKER, player=PlayerRole.DEFENDER
    )
    assert not res

    # None, Attacker
    res = Hnefatafl.piece_belongs_to_player(piece=None, player=PlayerRole.ATTACKER)
    assert not res

    # None, Defender
    res = Hnefatafl.piece_belongs_to_player(piece=None, player=PlayerRole.DEFENDER)
    assert not res

    # King, Attacker
    res = Hnefatafl.piece_belongs_to_player(
        piece=PieceType.KING, player=PlayerRole.ATTACKER
    )
    assert not res

    # King, Defender
    res = Hnefatafl.piece_belongs_to_player(
        piece=PieceType.KING, player=PlayerRole.DEFENDER
    )
    assert res


def test_game_over():
    board = copy.deepcopy(default_board)

    # king reached corner square -> Defenders win
    corner = Position(0, 0)
    corner.set_square(board=board, piece=PieceType.KING)
    Position(3, 3).set_square(board=board, piece=None)
    res = Hnefatafl.game_over(
        board=board,
        player=PlayerRole.DEFENDER,
        attackers=[],
    )
    assert res[0] == GameResult.WIN and res[1] == PlayerRole.DEFENDER

    # king was captured -> Attackers win
    corner.set_square(board=board, piece=None)
    res = Hnefatafl.game_over(
        attackers=[Position(1, 1)],
        board=board,
        player=PlayerRole.ATTACKER,
    )
    assert res[0] == GameResult.WIN and res[1] == PlayerRole.ATTACKER

    # No possible moves
    # TODO

    # No attackers left
    # TODO


def test_is_opponent():
    # defender vs defender
    res = Hnefatafl.is_opponent(
        piece_type=PieceType.DEFENDER, of_player=PlayerRole.DEFENDER
    )
    assert not res

    # defender vs attacker
    res = Hnefatafl.is_opponent(
        piece_type=PieceType.ATTACKER, of_player=PlayerRole.DEFENDER
    )
    assert res

    # defender vs king
    res = Hnefatafl.is_opponent(
        piece_type=PieceType.KING, of_player=PlayerRole.DEFENDER
    )
    assert not res

    # defender vs None
    res = Hnefatafl.is_opponent(
        piece_type=None,
        of_player=PlayerRole.DEFENDER,
    )
    assert not res

    # attacker vs attacker
    res = Hnefatafl.is_opponent(
        piece_type=PieceType.ATTACKER, of_player=PlayerRole.ATTACKER
    )
    assert not res

    # attacker vs defender
    res = Hnefatafl.is_opponent(
        piece_type=PieceType.DEFENDER, of_player=PlayerRole.ATTACKER
    )
    assert res

    # attacker vs king
    res = Hnefatafl.is_opponent(
        piece_type=PieceType.KING, of_player=PlayerRole.ATTACKER
    )
    assert res

    # attacker vs None
    res = Hnefatafl.is_opponent(piece_type=None, of_player=PlayerRole.ATTACKER)
    assert not res


def test_get_rendering_string():
    board = copy.deepcopy(default_board)
    res = Hnefatafl.get_rendering_string(board=board)
    print(f"at (3,3) is: {Position(3,3).get_square(board=board)}")
    assert (
        res
        == "  A   B   C   D   E   F   G\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 7\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 6\n+---+---+---+---+---+---+---+\n|   |   |   | 🛡️ |   |   |   | 5\n+---+---+---+---+---+---+---+\n| 🗡️ | 🗡️ | 🛡️ | K | 🛡️ | 🗡️ | 🗡️ | 4\n+---+---+---+---+---+---+---+\n|   |   |   | 🛡️ |   |   |   | 3\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 2\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 1\n+---+---+---+---+---+---+---+\n"
    )


def test_piece_captured():
    # DAD
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.ATTACKER)
    pos.left().set_square(board=empty, piece=PieceType.DEFENDER)
    pos.right().set_square(board=empty, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.left(),
        maybe_captured=pos,
        other_side=pos.right(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # ADA
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.DEFENDER)
    pos.left().set_square(board=empty, piece=PieceType.ATTACKER)
    pos.right().set_square(board=empty, piece=PieceType.ATTACKER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.left(),
        maybe_captured=pos,
        other_side=pos.right(),
        player=PlayerRole.ATTACKER,
        board=empty,
    )

    # KAD
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.ATTACKER)
    pos.left().set_square(board=empty, piece=PieceType.KING)
    pos.right().set_square(board=empty, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.left(),
        maybe_captured=pos,
        other_side=pos.right(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # KDA
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.DEFENDER)
    pos.left().set_square(board=empty, piece=PieceType.KING)
    pos.right().set_square(board=empty, piece=PieceType.ATTACKER)
    assert not Hnefatafl.piece_captured(
        new_pos=pos.left(),
        maybe_captured=pos,
        other_side=pos.right(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # D
    # A
    # D
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.ATTACKER)
    pos.up().set_square(board=empty, piece=PieceType.DEFENDER)
    pos.down().set_square(board=empty, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.up(),
        maybe_captured=pos,
        other_side=pos.down(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # A
    # D
    # A
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.DEFENDER)
    pos.up().set_square(board=empty, piece=PieceType.ATTACKER)
    pos.down().set_square(board=empty, piece=PieceType.ATTACKER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.up(),
        maybe_captured=pos,
        other_side=pos.down(),
        player=PlayerRole.ATTACKER,
        board=empty,
    )

    # K
    # D
    # A
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.DEFENDER)
    pos.up().set_square(board=empty, piece=PieceType.KING)
    pos.down().set_square(board=empty, piece=PieceType.ATTACKER)
    assert not Hnefatafl.piece_captured(
        new_pos=pos.up(),
        maybe_captured=pos,
        other_side=pos.down(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # K
    # A
    # D
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.ATTACKER)
    pos.up().set_square(board=empty, piece=PieceType.KING)
    pos.down().set_square(board=empty, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.up(),
        maybe_captured=pos,
        other_side=pos.down(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # RDA (R = restricted field)
    empty = copy.deepcopy(empty_board)
    pos = Position(y=0, x=1)
    pos.set_square(board=empty, piece=PieceType.DEFENDER)
    pos.right().set_square(board=empty, piece=PieceType.ATTACKER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.right(),
        maybe_captured=pos,
        other_side=pos.left(),
        player=PlayerRole.ATTACKER,
        board=empty,
    )

    # RAD
    empty = copy.deepcopy(empty_board)
    pos = Position(y=0, x=1)
    pos.set_square(board=empty, piece=PieceType.ATTACKER)
    pos.right().set_square(board=empty, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.right(),
        maybe_captured=pos,
        other_side=pos.left(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # R
    # D
    # A
    empty = copy.deepcopy(empty_board)
    pos = Position(y=Hnefatafl.DIMENSION - 2, x=0)
    pos.set_square(board=empty, piece=PieceType.DEFENDER)
    pos.down().set_square(board=empty, piece=PieceType.ATTACKER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.down(),
        maybe_captured=pos,
        other_side=pos.up(),
        player=PlayerRole.ATTACKER,
        board=empty,
    )

    # R
    # A
    # D
    empty = copy.deepcopy(empty_board)
    pos = Position(y=Hnefatafl.DIMENSION - 2, x=0)
    pos.set_square(board=empty, piece=PieceType.ATTACKER)
    pos.down().set_square(board=empty, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.down(),
        maybe_captured=pos,
        other_side=pos.up(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # EDA (E = empty throne)
    empty = copy.deepcopy(empty_board)
    pos = Position(y=3, x=4)
    pos.set_square(board=empty, piece=PieceType.DEFENDER)
    pos.right().set_square(board=empty, piece=PieceType.ATTACKER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.right(),
        maybe_captured=pos,
        other_side=pos.left(),
        player=PlayerRole.ATTACKER,
        board=empty,
    )

    # EAD (E = empty throne)
    empty = copy.deepcopy(empty_board)
    pos = Position(y=3, x=4)
    pos.set_square(board=empty, piece=PieceType.ATTACKER)
    pos.right().set_square(board=empty, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.right(),
        maybe_captured=pos,
        other_side=pos.left(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # E
    # D
    # A
    empty = copy.deepcopy(empty_board)
    pos = Position(y=2, x=3)
    pos.set_square(board=empty, piece=PieceType.DEFENDER)
    pos.down().set_square(board=empty, piece=PieceType.ATTACKER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.down(),
        maybe_captured=pos,
        other_side=pos.up(),
        player=PlayerRole.ATTACKER,
        board=empty,
    )

    # E
    # A
    # D
    empty = copy.deepcopy(empty_board)
    pos = Position(y=2, x=3)
    pos.set_square(board=empty, piece=PieceType.ATTACKER)
    pos.down().set_square(board=empty, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.down(),
        maybe_captured=pos,
        other_side=pos.up(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # ODA (O = occupied throne)
    empty = copy.deepcopy(empty_board)
    Position(3, 3).set_square(board=empty, piece=PieceType.KING)
    pos = Position(y=3, x=4)
    pos.set_square(board=empty, piece=PieceType.DEFENDER)
    pos.right().set_square(board=empty, piece=PieceType.ATTACKER)
    assert not Hnefatafl.piece_captured(
        new_pos=pos.right(),
        maybe_captured=pos,
        other_side=pos.left(),
        player=PlayerRole.ATTACKER,
        board=empty,
    )

    # OAD (O = occupied throne)
    empty = copy.deepcopy(empty_board)
    Position(3, 3).set_square(board=empty, piece=PieceType.KING)
    pos = Position(y=3, x=4)
    pos.set_square(board=empty, piece=PieceType.ATTACKER)
    pos.right().set_square(board=empty, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.right(),
        maybe_captured=pos,
        other_side=pos.left(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # O
    # D
    # A
    empty = copy.deepcopy(empty_board)
    Position(3, 3).set_square(board=empty, piece=PieceType.KING)
    pos = Position(y=2, x=3)
    pos.set_square(board=empty, piece=PieceType.DEFENDER)
    pos.down().set_square(board=empty, piece=PieceType.ATTACKER)
    assert not Hnefatafl.piece_captured(
        new_pos=pos.down(),
        maybe_captured=pos,
        other_side=pos.up(),
        player=PlayerRole.ATTACKER,
        board=empty,
    )

    # O
    # A
    # D
    empty = copy.deepcopy(empty_board)
    Position(3, 3).set_square(board=empty, piece=PieceType.KING)
    pos = Position(y=2, x=3)
    pos.set_square(board=empty, piece=PieceType.ATTACKER)
    pos.down().set_square(board=empty, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=pos.down(),
        maybe_captured=pos,
        other_side=pos.up(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # AKA
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.KING)
    pos.left().set_square(board=empty, piece=PieceType.ATTACKER)
    pos.right().set_square(board=empty, piece=PieceType.ATTACKER)
    assert not Hnefatafl.piece_captured(  # should not capture king, because capturing of king is handled in king_captured
        new_pos=pos.left(),
        maybe_captured=pos,
        other_side=pos.right(),
        player=PlayerRole.ATTACKER,
        board=empty,
    )

    # A
    # K
    # A
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.KING)
    pos.up().set_square(board=empty, piece=PieceType.ATTACKER)
    pos.down().set_square(board=empty, piece=PieceType.ATTACKER)
    assert not Hnefatafl.piece_captured(  # should not capture king, because capturing of king is handled in king_captured
        new_pos=pos.up(),
        maybe_captured=pos,
        other_side=pos.down(),
        player=PlayerRole.ATTACKER,
        board=empty,
    )

    # DKD
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.KING)
    pos.left().set_square(board=empty, piece=PieceType.DEFENDER)
    pos.right().set_square(board=empty, piece=PieceType.DEFENDER)
    assert not Hnefatafl.piece_captured(
        new_pos=pos.left(),
        maybe_captured=pos,
        other_side=pos.right(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )

    # D
    # K
    # D
    empty = copy.deepcopy(empty_board)
    pos = Position(2, 1)
    pos.set_square(board=empty, piece=PieceType.KING)
    pos.up().set_square(board=empty, piece=PieceType.DEFENDER)
    pos.down().set_square(board=empty, piece=PieceType.DEFENDER)
    assert not Hnefatafl.piece_captured(
        new_pos=pos.up(),
        maybe_captured=pos,
        other_side=pos.down(),
        player=PlayerRole.DEFENDER,
        board=empty,
    )


def test_piece_captures_opponent():
    # TODO
    pass


def test_king_is_captured():
    # king is on throne and surrounded by attackers
    middle_captured = copy.deepcopy(default_board)
    Position(y=2, x=3).set_square(board=middle_captured, piece=PieceType.ATTACKER)
    Position(y=3, x=2).set_square(board=middle_captured, piece=PieceType.ATTACKER)
    Position(y=3, x=4).set_square(board=middle_captured, piece=PieceType.ATTACKER)
    Position(y=4, x=3).set_square(board=middle_captured, piece=PieceType.ATTACKER)
    assert Hnefatafl.king_is_captured(
        board=middle_captured,
        new_pos=Position(3, 2),
        player=PlayerRole.ATTACKER,
    )

    # king is on throne, but not fully surrounded by attackers
    Position(y=4, x=3).set_square(board=middle_captured, piece=None)
    assert not Hnefatafl.king_is_captured(
        board=middle_captured,
        new_pos=Position(3, 2),
        player=PlayerRole.ATTACKER,
    )

    # king is next to throne and surrounded by attackers on other 3 sides
    next_to_throne_captured = copy.deepcopy(empty_board)
    king_pos = Position(y=4, x=3)
    king_pos.set_square(board=next_to_throne_captured, piece=PieceType.KING)
    king_pos.up().set_square(board=next_to_throne_captured, piece=PieceType.ATTACKER)
    king_pos.left().set_square(board=next_to_throne_captured, piece=PieceType.ATTACKER)
    king_pos.right().set_square(board=next_to_throne_captured, piece=PieceType.ATTACKER)
    assert Hnefatafl.king_is_captured(
        board=next_to_throne_captured,
        new_pos=king_pos.up(),
        player=PlayerRole.ATTACKER,
    )

    # king is on "random" square (not restricted, not throne, not next to throne) and captured
    # AKA
    random_captured = copy.deepcopy(empty_board)
    king_pos = Position(2, 1)
    king_pos.set_square(board=random_captured, piece=PieceType.KING)
    king_pos.left().set_square(board=random_captured, piece=PieceType.ATTACKER)
    king_pos.right().set_square(board=random_captured, piece=PieceType.ATTACKER)
    assert Hnefatafl.king_is_captured(
        board=random_captured,
        new_pos=king_pos.left(),
        player=PlayerRole.ATTACKER,
    )

    # A
    # K
    # A
    random_captured = copy.deepcopy(empty_board)
    king_pos = Position(2, 1)
    king_pos.set_square(board=random_captured, piece=PieceType.KING)
    king_pos.up().set_square(board=random_captured, piece=PieceType.ATTACKER)
    king_pos.down().set_square(board=random_captured, piece=PieceType.ATTACKER)
    assert Hnefatafl.king_is_captured(
        board=random_captured,
        new_pos=king_pos.up(),
        player=PlayerRole.ATTACKER,
    )
    # DKD
    random_captured = copy.deepcopy(empty_board)
    king_pos = Position(2, 1)
    king_pos.set_square(board=random_captured, piece=PieceType.KING)
    king_pos.left().set_square(board=random_captured, piece=PieceType.DEFENDER)
    king_pos.right().set_square(board=random_captured, piece=PieceType.DEFENDER)
    assert not Hnefatafl.king_is_captured(
        board=random_captured,
        new_pos=king_pos.left(),
        player=PlayerRole.DEFENDER,
    )
    # DKA
    random_captured = copy.deepcopy(empty_board)
    king_pos = Position(2, 1)
    king_pos.set_square(board=random_captured, piece=PieceType.KING)
    king_pos.left().set_square(board=random_captured, piece=PieceType.DEFENDER)
    king_pos.right().set_square(board=random_captured, piece=PieceType.ATTACKER)
    assert not Hnefatafl.king_is_captured(
        board=random_captured,
        new_pos=king_pos.right(),
        player=PlayerRole.ATTACKER,
    )


def test_my_step():
    # TODO
    pass


def test_action_to_move():
    hnefatafl = Hnefatafl()
    start_pos, end_pos = hnefatafl.action_to_move(766)
    assert start_pos == Position(y=1, x=2) and end_pos == Position(y=3, x=4)


def test_move_to_action():
    hnefatafl = Hnefatafl()
    start_pos = Position(y=1, x=2)
    end_pos = Position(y=3, x=4)
    move = (start_pos, end_pos)
    assert hnefatafl.move_to_action(move) == 766


def test_get_observation():
    board = copy.deepcopy(default_board)
    obs = Hnefatafl.get_observation(board=board)
    assert len(obs.shape) == 3
    attackers = Hnefatafl.get_attackers(board=board)
    defenders = Hnefatafl.get_defenders(board=board)
    king = Hnefatafl.get_king(board=board)
    for i in range(Hnefatafl.DIMENSION):
        for j in range(Hnefatafl.DIMENSION):
            pos = Position(i, j)
            if pos in attackers:
                assert obs[0, i, j] == 1
            if pos in defenders:
                assert obs[1, i, j] == 1
            if pos == king:
                assert obs[2, i, j] == 1
