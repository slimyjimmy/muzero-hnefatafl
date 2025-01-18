from enum import Enum


class PieceType(Enum):
    DEFENDER = 0
    ATTACKER = 1
    KING = 2

    def to_string(self) -> str:
        if self == PieceType.DEFENDER:
            return "🛡️"
        if self == PieceType.ATTACKER:
            return "🗡️"
        if self == PieceType.KING:
            return "K"
