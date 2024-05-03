"""Submodule for main logic implementation(servers).

API for client-server interaction provided.
"""

import dataclasses
import datetime
import enum


class Num(enum.IntEnum):
    """Subtype of int.

    Representing 3 entites:
    1) Possible numbers in sudoku
    2) Possible row indexes
    3) Possible column indexes
    """

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
    """Type representing point on board."""

    x: Num
    y: Num


class SudokuSession:
    """Representation of one game.

    Declares API for initializing the game,
    getting information about board and making step.
    """

    def __init__(self) -> None:
        """Game initializer.

        Computes random sudoku board, ready to game.
        """
        raise NotImplementedError

    def undo(self) -> bool:
        """Undo last turn.

        Returns: true if the turn was successful, false otherwise
        """
        raise NotImplementedError

    def redo(self) -> bool:
        """Redo last turn.

        Returns: true if the turn was successful, false otherwise
        """
        raise NotImplementedError

    def set_num(self, pos: Pos, num: Num) -> bool:
        """Set the point at 'pos' value to 'num'.

        Returns: true if the turn was successful, false otherwise
        """
        raise NotImplementedError

    def del_num(self, pos: Pos) -> bool:
        """Unset the point at 'pos'.

        Returns: true if the turn was successful, false otherwise
        """
        raise NotImplementedError

    def get_errors(self) -> list[list[bool]]:
        """Get matrix of errors on board."""
        raise NotImplementedError

    def get_initials(self) -> list[list[bool]]:
        """Get matrix of initials on board."""
        raise NotImplementedError

    def get_board(self) -> list[list[Num | None]]:
        """Get matrix of board values."""
        raise NotImplementedError


class Difficulty(enum.Enum):
    """Difficulty of game."""

    Easy = 0
    Medium = 1
    Hard = 2


@dataclasses.dataclass
class Save:
    """Representation of save metadata."""

    name: str
    timestamp: datetime.datetime
    difficulty: Difficulty
    session_filename: str


class SudokuServer:
    """Representation of sudoku server.

    Implements all server-side logic except game managing itself.
    """

    _saves_directory: str

    def __init__(self) -> None:
        """Server initializer."""
        raise NotImplementedError

    def generate_session(self, difficulty: Difficulty = Difficulty.Medium) -> SudokuSession:
        """Generate session.

        Typically used in 'New Game'.
        """
        raise NotImplementedError

    def save_session(self, session: SudokuSession) -> None:
        """Save session for  further gaming.

        Typically used in 'Save'.
        """
        raise NotImplementedError

    def list_saves(self) -> list[Save]:
        """List saves metadata for further loading.

        Typically used in some kind of 'Continue' menu.
        """
        raise NotImplementedError

    def load_session(self, save: Save) -> SudokuSession:
        """Load session from save.

        Typically used in some kind of 'Continue' menu.
        """
        raise NotImplementedError
