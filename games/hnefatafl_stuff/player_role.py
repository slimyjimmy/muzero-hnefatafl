from enum import Enum


class PlayerRole(Enum):
    DEFENDER = 1
    ATTACKER = -1

    def to_string(self) -> str:
        if self == PlayerRole.DEFENDER:
            return "Defender"
        return "Attacker"

    def toggle(self):
        return PlayerRole(self.value * -1)
