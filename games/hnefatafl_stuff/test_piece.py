from games.hnefatafl_stuff.piece import Piece
from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.position import Position


def test_init():
    pos = Position(x=1, y=1)
    piece = Piece(piece_type=PieceType.KING, position=pos)
    assert piece.piece_type == PieceType.KING
    assert piece.position == pos
