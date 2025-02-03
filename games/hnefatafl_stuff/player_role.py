from enum import Enum


class PlayerRole(Enum):
    DEFENDER = "Defender"
    ATTACKER = "Attacker"

    def to_string(self) -> str:
        if self == PlayerRole.DEFENDER:
            return "Defender"
        return "Attacker"

    def toggle(self):
        if self == PlayerRole.DEFENDER:
            return PlayerRole.ATTACKER
        return PlayerRole.DEFENDER
