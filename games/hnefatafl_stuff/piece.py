from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.position import Position


class Piece:
    def __init__(self, piece_type: PieceType, position: Position):
        self.piece_type = piece_type
        self.position = position
        self.captured = False
