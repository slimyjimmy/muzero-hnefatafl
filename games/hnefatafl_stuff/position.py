from pydantic import BaseModel
from typing import ClassVar, List, Optional
from games.hnefatafl_stuff.direction import Direction
from games.hnefatafl_stuff.piece_type import PieceType


class Position(BaseModel):
    """
    Represents a zero-based position on the board.
    """

    x: int
    y: int

    INVALID: ClassVar[int] = -1

    def get_adjacent_position(self, direction: Direction, steps: int = 1) -> "Position":
        if direction == Direction.UP:
            return self.up(steps=steps)
        if direction == Direction.DOWN:
            return self.down(steps=steps)
        if direction == Direction.LEFT:
            return self.left(steps=steps)
        if direction == Direction.RIGHT:
            return self.right(steps=steps)
        return None

    def is_adjacent(self, other) -> Optional[Direction]:
        """
        Checks if self and other are adjacent to each other.
        If so: returns the relative direction of other to self (ie: if other is above self -> Direction.UP)
        Else: returns None
        """
        if self.up() == other:
            return Direction.UP
        if self.down() == other:
            return Direction.DOWN
        if self.right() == other:
            return Direction.RIGHT
        if self.left() == other:
            return Direction.LEFT
        return None

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def up(self, steps: int = 1) -> "Position":
        return Position(x=self.x, y=self.y + steps)

    def left(self, steps: int = 1) -> "Position":
        return Position(x=self.x - steps, y=self.y)

    def right(self, steps: int = 1) -> "Position":
        return Position(x=self.x + steps, y=self.y)

    def down(self, steps: int = 1) -> "Position":
        return Position(x=self.x, y=self.y - steps)

    def get_square(self, board: List[List[Optional[PieceType]]]) -> Optional[PieceType]:
        if self.x < 0 or self.x >= len(board) or self.y < 0 or self.y >= len(board):
            return None
        return board[self.y][self.x]

    def set_square(
        self, board: List[List[Optional[PieceType]]], piece: Optional[PieceType]
    ) -> None:
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

    def to_list(self) -> List[int]:
        return [self.x, self.y]
