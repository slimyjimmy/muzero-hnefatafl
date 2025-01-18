from typing import List, Optional
import unittest

from games.hnefatafl_stuff.hnefatafl import Hnefatafl
from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.position import Position
from games.hnefatafl_stuff.types import Board


class TestHnefatafl(unittest.TestCase):

    def test_get_attackers(self):
        raise NotImplementedError("This method is not implemented yet.")

    def test_get_defenders(self):
        raise NotImplementedError("This method is not implemented yet.")

    def test_get_possible_dests_from_pos(self):
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
        self.assertEqual(len(res), (7 - 3) + (7 - 1))

    def test_get_possible_moves(self):
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
        res = hnefatafl.get_possible_moves(board=board_1)
        self.assertEqual(len(res), (7 - 3) + (7 - 1))

    def test_piece_belongs_to_player(self):
        raise NotImplementedError("This method is not implemented yet.")

    def test_game_over(self):
        raise NotImplementedError("This method is not implemented yet.")

    def test_is_opponent(self):
        raise NotImplementedError("This method is not implemented yet.")

    def test_get_rendering_string(self):
        raise NotImplementedError("This method is not implemented yet.")
