from typing import List, Optional
from games.hnefatafl_stuff.piece_type import PieceType


class Position:
    """
    Represents a zero-based position on the board.
    """

    INVALID = -1

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def up(self, steps: int = 1) -> "Position":
        return Position(self.x, self.y + steps)

    def left(self, steps: int = 1) -> "Position":
        return Position(self.x - steps, self.y)

    def right(self, steps: int = 1) -> "Position":
        return Position(self.x + steps, self.y)

    def down(self, steps: int = 1) -> "Position":
        return Position(self.x, self.y - steps)

    def get_square(self, board: List[List[Optional[PieceType]]]) -> Optional[PieceType]:
        if self.x < 0 or self.x >= len(board) or self.y < 0 or self.y >= len(board):
            return None
        return board[self.y][self.x]

    def set_square(
        self, board: List[List[Optional[PieceType]]], piece: Optional[PieceType]
    ):
        board[self.y][self.x] = piece

    def is_within_board(self) -> bool:
        """Checks if a position is on the board"""
        from games.hnefatafl_stuff.hnefatafl import Hnefatafl

        return 0 <= self.x < Hnefatafl.DIMENSION and 0 <= self.y < Hnefatafl.DIMENSION

    def is_open_to_piece(self, piece: PieceType) -> bool:
        """
        Checks if a position is
            - on the board
            - not restricted for piece (only king can enter restricted positions)
        Returns true if position is open to given piece and false otherwise.
        """
        from games.hnefatafl_stuff.hnefatafl import Hnefatafl

        if piece is None:
            print("Piece is none")
            return False

        if not self.is_within_board():
            return False
        if self in Hnefatafl.RESTRICTED_POSITIONS:
            return piece == PieceType.KING  # only king can enter restricted squares
        return True

    def to_string(self) -> str:
        from games.hnefatafl_stuff.hnefatafl import Hnefatafl

        return f"({chr(ord('A') + self.x)}{abs(self.y - Hnefatafl.DIMENSION)})"
