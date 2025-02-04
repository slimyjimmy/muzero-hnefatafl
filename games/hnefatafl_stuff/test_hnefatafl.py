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
    corner = Position(x=0, y=0)
    corner.set_square(board=board, piece=PieceType.KING)
    Position(x=3, y=3).set_square(board=board, piece=None)
    res = Hnefatafl.game_over(
        board=board,
        attackers=[],
        player=PlayerRole.ATTACKER,
    )
    assert res[0] == GameResult.WIN and res[1] == PlayerRole.DEFENDER

    # king was captured -> Attackers win
    corner.set_square(board=board, piece=None)
    res = Hnefatafl.game_over(
        attackers=[Position(x=1, y=1)],
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
    print(f"at (3,3) is: {Position(x=3,y=3).get_square(board=board)}")
    assert (
        res
        == "  A   B   C   D   E   F   G\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 7\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 6\n+---+---+---+---+---+---+---+\n|   |   |   | 🛡️ |   |   |   | 5\n+---+---+---+---+---+---+---+\n| 🗡️ | 🗡️ | 🛡️ | K | 🛡️ | 🗡️ | 🗡️ | 4\n+---+---+---+---+---+---+---+\n|   |   |   | 🛡️ |   |   |   | 3\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 2\n+---+---+---+---+---+---+---+\n|   |   |   | 🗡️ |   |   |   | 1\n+---+---+---+---+---+---+---+\n"
    )


def test_piece_captured():
    # DAD
    empty = copy.deepcopy(empty_board)
    pos = Position(x=2, y=1)
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
    pos = Position(x=2, y=1)
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
    pos = Position(x=2, y=1)
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
    pos = Position(x=2, y=1)
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
    pos = Position(x=2, y=1)
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
    pos = Position(x=2, y=1)
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
    pos = Position(x=2, y=1)
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
    pos = Position(x=2, y=1)
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
    Position(x=3, y=3).set_square(board=empty, piece=PieceType.KING)
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
    Position(x=3, y=3).set_square(board=empty, piece=PieceType.KING)
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
    Position(x=3, y=3).set_square(board=empty, piece=PieceType.KING)
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
    Position(x=3, y=3).set_square(board=empty, piece=PieceType.KING)
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
    pos = Position(x=2, y=1)
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
    pos = Position(x=2, y=1)
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
    pos = Position(x=2, y=1)
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
    pos = Position(x=2, y=1)
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
        new_pos=Position(x=3, y=2),
        player=PlayerRole.ATTACKER,
    )

    # king is on throne, but not fully surrounded by attackers
    Position(y=4, x=3).set_square(board=middle_captured, piece=None)
    assert not Hnefatafl.king_is_captured(
        board=middle_captured,
        new_pos=Position(x=3, y=2),
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
    king_pos = Position(x=2, y=1)
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
    king_pos = Position(x=2, y=1)
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
    king_pos = Position(x=2, y=1)
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
    king_pos = Position(x=2, y=1)
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
            pos = Position(x=i, y=j)
            if pos in attackers:
                assert obs[0, i, j] == 1
            if pos in defenders:
                assert obs[1, i, j] == 1
            if pos == king:
                assert obs[2, i, j] == 1


def test_get_board_from_observation():
    og_board = copy.deepcopy(default_board)
    observation = Hnefatafl.get_observation(board=og_board)
    board_from_obs = Hnefatafl.get_board_from_observation(observation=observation)
    for i in range(Hnefatafl.DIMENSION):
        for j in range(Hnefatafl.DIMENSION):
            pos = Position(x=i, y=j)
            assert pos.get_square(board=og_board) == pos.get_square(
                board=board_from_obs
            )


# Rules from https://aagenielsen.dk/copenhagen_rules.php
def test_1():
    board = copy.deepcopy(Hnefatafl.DEFAULT_BOARD)
    # there are twice as many attackers as defenders
    attackers = Hnefatafl.get_attackers(board=board)
    defenders = Hnefatafl.get_defenders(board=board)
    assert len(attackers) == 2 * len(defenders)


def test_2():
    # it's attacker's turn first
    hnef = Hnefatafl()
    assert hnef.current_player == PlayerRole.ATTACKER


def test_3():
    middle = Position(x=3, y=3)

    # test attacker
    board = copy.deepcopy(empty_board)
    pos = middle.left()
    pos.set_square(board=board, piece=PieceType.ATTACKER)
    print(Hnefatafl.get_rendering_string(board=board))
    possible_dests = Hnefatafl.get_possible_dests_from_pos(
        board=board, player=PlayerRole.ATTACKER, start_pos=pos
    )
    assert len(possible_dests) == 5 + 6
    for possible_dest in possible_dests:
        assert (
            possible_dest.x == pos.x or possible_dest.y == pos.y
        ), possible_dest.to_string()

    # test defender
    board = copy.deepcopy(empty_board)
    pos = middle.left()
    pos.set_square(board=board, piece=PieceType.DEFENDER)
    possible_dests = Hnefatafl.get_possible_dests_from_pos(
        board=board, player=PlayerRole.DEFENDER, start_pos=pos
    )
    assert len(possible_dests) == 5 + 6
    for possible_dest in possible_dests:
        assert (
            possible_dest.x == pos.x or possible_dest.y == pos.y
        ), possible_dest.to_string()

    # test king
    board = copy.deepcopy(empty_board)
    middle.set_square(board=board, piece=PieceType.KING)
    possible_dests = Hnefatafl.get_possible_dests_from_pos(
        board=board, player=PlayerRole.DEFENDER, start_pos=middle
    )
    assert len(possible_dests) == 6 + 6
    for possible_dest in possible_dests:
        assert (
            possible_dest.x == middle.x or possible_dest.y == middle.y
        ), possible_dest.to_string()


def test_4a():
    # pic 1 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    board = copy.deepcopy(empty_board)
    Position(x=0, y=1).set_square(board=board, piece=PieceType.ATTACKER)
    Position(x=2, y=3).set_square(board=board, piece=PieceType.ATTACKER)
    Position(x=4, y=1).set_square(board=board, piece=PieceType.ATTACKER)
    Position(x=1, y=1).set_square(board=board, piece=PieceType.DEFENDER)
    Position(x=2, y=2).set_square(board=board, piece=PieceType.DEFENDER)
    Position(x=3, y=1).set_square(board=board, piece=PieceType.DEFENDER)
    new_pos = Position(x=2, y=1)
    new_pos.set_square(board=board, piece=PieceType.ATTACKER)
    assert Hnefatafl.piece_captured(
        new_pos=new_pos,
        maybe_captured=new_pos.left(),
        other_side=new_pos.left(2),
        player=PlayerRole.ATTACKER,
        board=board,
    )
    assert Hnefatafl.piece_captured(
        new_pos=new_pos,
        maybe_captured=new_pos.right(),
        other_side=new_pos.right(2),
        player=PlayerRole.ATTACKER,
        board=board,
    )
    assert Hnefatafl.piece_captured(
        new_pos=new_pos,
        maybe_captured=new_pos.up(),
        other_side=new_pos.up(2),
        player=PlayerRole.ATTACKER,
        board=board,
    )

    # pic 2 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    board = copy.deepcopy(empty_board)
    middle = Position(x=3, y=3)
    middle.down().set_square(board=board, piece=PieceType.DEFENDER)
    middle.down(2).set_square(board=board, piece=PieceType.ATTACKER)
    assert Hnefatafl.piece_captured(
        new_pos=middle.down(2),
        maybe_captured=middle.down(),
        other_side=middle,
        player=PlayerRole.ATTACKER,
        board=board,
    )

    # pic 3 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    board = copy.deepcopy(empty_board)
    white_pos = Position(x=2, y=1)
    white_pos.set_square(board=board, piece=PieceType.DEFENDER)
    white_pos.up().set_square(board=board, piece=PieceType.ATTACKER)
    white_pos.up(2).set_square(board=board, piece=PieceType.KING)
    assert Hnefatafl.piece_captured(
        new_pos=white_pos.up(2),
        maybe_captured=white_pos.up(),
        other_side=white_pos,
        player=PlayerRole.DEFENDER,
        board=board,
    )

    # pic 4 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    board = copy.deepcopy(empty_board)
    middle = Position(x=3, y=3)
    middle.set_square(board=board, piece=PieceType.KING)
    middle.down().set_square(board=board, piece=PieceType.ATTACKER)
    middle.down(2).set_square(board=board, piece=PieceType.DEFENDER)
    assert Hnefatafl.piece_captured(
        new_pos=middle.down(2),
        maybe_captured=middle.down(),
        other_side=middle,
        player=PlayerRole.DEFENDER,
        board=board,
    )

    # pic 5 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    board = copy.deepcopy(empty_board)
    white_pos = Position(x=1, y=6)
    white_pos.set_square(board=board, piece=PieceType.DEFENDER)
    white_pos.right().set_square(board=board, piece=PieceType.ATTACKER)
    assert Hnefatafl.piece_captured(
        new_pos=white_pos.right(),
        maybe_captured=white_pos,
        other_side=white_pos.left(),
        player=PlayerRole.ATTACKER,
        board=board,
    )

    # pic 6 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    board = copy.deepcopy(empty_board)
    middle = Position(x=3, y=3)
    middle.set_square(board=board, piece=PieceType.KING)
    middle.down().set_square(board=board, piece=PieceType.DEFENDER)
    middle.down(2).set_square(board=board, piece=PieceType.ATTACKER)
    assert not Hnefatafl.piece_captured(
        new_pos=middle.down(2),
        maybe_captured=middle.down(),
        other_side=middle,
        player=PlayerRole.ATTACKER,
        board=board,
    )

    # pic 7 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    assert True


def test_4b():
    # we don't use this rule (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    pass


def test_5():
    for corner in Hnefatafl.CORNERS:
        assert not corner.is_open_to_piece(piece=PieceType.ATTACKER)
        assert not corner.is_open_to_piece(piece=PieceType.DEFENDER)
        assert corner.is_open_to_piece(piece=PieceType.KING)

    assert not Hnefatafl.MIDDLE.is_open_to_piece(piece=PieceType.ATTACKER)
    assert not Hnefatafl.MIDDLE.is_open_to_piece(piece=PieceType.DEFENDER)
    assert Hnefatafl.MIDDLE.is_open_to_piece(piece=PieceType.KING)


def test_6a():
    for corner in Hnefatafl.CORNERS:
        board = copy.deepcopy(empty_board)
        corner.set_square(board=board, piece=PieceType.KING)
        Position(x=1, y=1).set_square(board=board, piece=PieceType.ATTACKER)
        res = Hnefatafl.game_over(
            board=board,
            attackers=[Position(x=1, y=1)],
            player=PlayerRole.ATTACKER,
        )
        assert res[0] == GameResult.WIN and res[1] == PlayerRole.DEFENDER


def test_6b():
    # we don't use this rule (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    pass


def test_7a():
    # explicit changes from Prof Koch:
    # - If the king is not at or next to the throne, he can be captured like any other piece, with two enemies at the sides
    # - The corner fields are hostile to all, including the King.

    # pic 1 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    board = copy.deepcopy(empty_board)
    Hnefatafl.MIDDLE.left().set_square(board=board, piece=PieceType.ATTACKER)
    Hnefatafl.MIDDLE.right().set_square(board=board, piece=PieceType.ATTACKER)
    Hnefatafl.MIDDLE.up().set_square(board=board, piece=PieceType.ATTACKER)
    Hnefatafl.MIDDLE.down().set_square(board=board, piece=PieceType.ATTACKER)
    res = Hnefatafl.game_over(
        board=board,
        attackers=Hnefatafl.get_attackers(board=board),
        player=PlayerRole.ATTACKER,
    )
    assert res[0] == GameResult.WIN and res[1] == PlayerRole.ATTACKER

    # pic 2 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    board = copy.deepcopy(empty_board)
    king_pos = Hnefatafl.MIDDLE.down()
    king_pos.left().set_square(board=board, piece=PieceType.ATTACKER)
    king_pos.right().set_square(board=board, piece=PieceType.ATTACKER)
    king_pos.down().set_square(board=board, piece=PieceType.ATTACKER)
    res = Hnefatafl.game_over(
        board=board,
        attackers=Hnefatafl.get_attackers(board=board),
        player=PlayerRole.ATTACKER,
    )
    assert res[0] == GameResult.WIN and res[1] == PlayerRole.ATTACKER

    # pic 3 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    # -> not necessary due to explicit changes from Prof Koch (see above)

    # pic 4 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    # -> not necessary due to explicit changes from Prof Koch (see above)

    # pic 5 (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    # -> not necessary due to explicit changes from Prof Koch (see above)


def test_7b():
    # we don't use this rule (see https://isis.tu-berlin.de/mod/forum/discuss.php?d=634272)
    assert True


def test_8():
    assert True  # max_moves set in hnefatafl_game controls this


def test_9():
    board = copy.deepcopy(empty_board)
    lower_attacker = Position(x=0, y=1)
    lower_attacker.set_square(board=board, piece=PieceType.ATTACKER)
    lower_attacker.up().set_square(board=board, piece=PieceType.ATTACKER)
    lower_attacker.up(2).set_square(board=board, piece=PieceType.DEFENDER)
    lower_attacker.up().right().set_square(board=board, piece=PieceType.DEFENDER)
    lower_attacker.right().set_square(board=board, piece=PieceType.DEFENDER)
    Hnefatafl.MIDDLE.set_square(board=board, piece=PieceType.KING)
    res = Hnefatafl.game_over(
        board=board,
        attackers=Hnefatafl.get_attackers(board=board),
        player=PlayerRole.ATTACKER,
    )
    assert res[0] == GameResult.WIN and res[1] == PlayerRole.DEFENDER


def test_10():
    assert True  # will lead to case 8
