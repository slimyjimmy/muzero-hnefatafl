from typing import List, Optional, Tuple

from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.position import Position


Move = Tuple[Position, Position]
Board = List[List[Optional[PieceType]]]
