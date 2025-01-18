from typing import List, Optional

from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.position import Position


class Hnefatafl_Rules:
    DIMENSION = 7

    def get_possible_destinations_from_pos(
        board: List[List[Optional[PieceType]]],
        start_pos: Position,
    ) -> List[Position]:
        return []
