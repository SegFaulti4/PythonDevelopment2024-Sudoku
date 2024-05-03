import enum
import dataclasses
import datetime


class Num(enum.IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9


@dataclasses.dataclass
class Pos:
    x: Num
    y: Num


class Session:
    def __init__(self) -> None:
        raise NotImplementedError

    def undo(self) -> bool:
        raise NotImplementedError

    def redo(self) -> bool:
        raise NotImplementedError

    def set_num(self, pos: Pos, num: Num) -> bool:
        raise NotImplementedError

    def del_num(self, pos: Pos) -> bool:
        raise NotImplementedError

    def get_errors(self) -> list[list[bool]]:
        raise NotImplementedError

    def get_initials(self) -> list[list[bool]]:
        raise NotImplementedError

    def get_board(self) -> list[list[Num]]:
        raise NotImplementedError


class Difficulty(enum.Enum):
    Easy = 0
    Medium = 1
    Hard = 2


@dataclasses.dataclass
class Save:
    name: str
    timestamp: datetime.datetime
    difficulty: Difficulty
    session_filename: str


class Game:
    _saves_directory: str

    def __init__(self) -> None:
        raise NotImplementedError

    def generate_session(self, difficulty: Difficulty = Difficulty.Medium) -> Session:
        raise NotImplementedError

    def save_session(self, session: Session) -> None:
        raise NotImplementedError

    def list_saves(self) -> list[Save]:
        raise NotImplementedError

    def load_session(self, save: Save) -> Session:
        raise NotImplementedError
