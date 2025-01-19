from typing import List, Optional, Tuple

from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.position import Position


Move = Tuple[Position, Position]
Board = List[List[Optional[PieceType]]]

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
