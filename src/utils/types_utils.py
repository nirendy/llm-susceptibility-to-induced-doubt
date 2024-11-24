from enum import Enum


class STREnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)
