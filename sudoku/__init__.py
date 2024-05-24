"""Submodule for main logic implementation(servers).

API for client-server interaction provided.
"""
from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import logging
import pathlib
import uuid
from itertools import permutations
from random import choice, sample
from typing import Any, Callable, Generic, TypeVar

import attrs
import attrs.validators
import cattrs
import cattrs.errors
import tomli
import tomli_w

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class SudokuError(Exception):
    """Base exception class for sudoku backend."""


class SudokuFileError(SudokuError):
    """Files related exception class."""

    path: pathlib.Path

    def __init__(self, *args: Any, path: pathlib.Path):
        """Create file error with path."""
        super().__init__(*args)
        self.path = path


class Num(enum.IntEnum):
    """Subtype of int.

    Representing 3 entities:
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


@attrs.define(slots=True)
class Pos:
    """Type representing point on board."""

    x: Num
    y: Num


class Difficulty(enum.Enum):
    """Difficulty of game."""

    Easy = 0
    Medium = 1
    Hard = 2


T = TypeVar("T")
Board = list[list[Num | None]]
FullBoard = list[list[Num]]
BoardMask = list[list[bool]]


@dataclasses.dataclass
class _LP(Generic[T]):
    pred: Callable[[T], bool] | None = None
    size: int = -1

    def __call__(self, value: list[T]) -> bool:
        if not isinstance(value, list):
            return False
        if self.size != -1 and len(value) != self.size:
            return False
        if self.pred is None:
            return True
        return all(self.pred(v) for v in value)


def _pv(pred: Callable[[T], bool]) -> Callable[[Any, Any, T], None]:
    def validator(_: Any, __: Any, value: T) -> None:
        if not pred(value):
            raise ValueError()

    return validator


_pred_9x9: Callable[[list[list]], bool] = _LP(size=9, pred=_LP(size=9))
_pred_list_9x9: Callable[[list[list[list]]], bool] = _LP(pred=_pred_9x9)


@attrs.define(slots=True, frozen=True)
class SessionSave:
    """Representation of session metadata."""

    name: str
    seed: int
    difficulty: Difficulty
    starting_field: Board = attrs.field(validator=[_pv(_pred_9x9)])
    session_id: str = str(uuid.UUID())
    timestamp: datetime.datetime = datetime.datetime.now()

    @staticmethod
    def _from_file(path: pathlib.Path) -> SessionSave:
        try:
            with open(path, "rb") as in_f:
                data = tomli.load(in_f)
            save = cattrs.structure(data, SessionSave)
            return save
        except cattrs.errors.ClassValidationError as exc:
            logger.warning(cattrs.transform_error(exc))
            raise SudokuFileError("Session metadata file corrupted", path=path)
        except Exception as exc:
            logger.warning(exc)
            raise SudokuFileError("Failed to load session metadata file", path=path)

    def _save_as_file(self, path: pathlib.Path) -> None:
        data = attrs.asdict(self)
        try:
            with open(path, "wb") as out_f:
                tomli_w.dump(data, out_f)
        except Exception as exc:
            logger.warning(exc)
            raise SudokuFileError("Failed to save session metadata file", path=path)


@attrs.define(slots=True)
class SessionHistory:
    """Representation of one game history."""

    full_board: FullBoard = attrs.field(validator=[_pv(_pred_9x9)])
    initial: BoardMask = attrs.field(validator=[_pv(_pred_9x9)])
    boards: list[Board] = attrs.field(validator=[_pv(_pred_list_9x9)])
    turn: int

    @staticmethod
    def _from_file(path: pathlib.Path) -> SessionHistory:
        try:
            with open(path, "r") as in_f:
                data = json.load(in_f)
            history = cattrs.structure(data, SessionHistory)
            return history
        except cattrs.errors.ClassValidationError as exc:
            logger.warning(cattrs.transform_error(exc))
            raise SudokuFileError("Save field file corrupted", path=path)
        except Exception as exc:
            logger.warning(exc)
            raise SudokuFileError("Failed to read field file", path=path)

    def _save_as_file(self, path: pathlib.Path) -> None:
        data = attrs.asdict(self)
        try:
            with open(path, "w") as out_f:
                json.dump(data, out_f)
        except Exception as exc:
            logger.warning(exc)
            raise SudokuFileError("Failed to save session history file", path=path)


class SudokuSession:
    """Representation of one game.

    Declares API for initializing the game,
    getting information about board and making step.
    """

    save: SessionSave
    history: SessionHistory

    def __init__(self, save: SessionSave, history: SessionHistory) -> None:
        """Game initializer.

        Computes random sudoku board, ready to game.
        """
        self.save = save
        self.history = history

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


class SudokuServer:
    """Representation of sudoku server.

    Implements all server-side logic except game managing itself.
    """

    _config_directory: pathlib.Path = pathlib.Path.home() / ".config" / "cmc_sudoku_2024"
    _saves_directory: pathlib.Path = _config_directory / "saves"

    def __init__(self) -> None:
        """Server initializer."""
        self._config_directory.mkdir(parents=True, exist_ok=True)
        self._saves_directory.mkdir(parents=True, exist_ok=True)

    def _session_history_path(self, session_id: str) -> pathlib.Path:
        return self._saves_directory / (session_id + ".json")

    def _session_save_path(self, session_id: str) -> pathlib.Path:
        return self._saves_directory / (session_id + ".toml")

    def _save_session_as_file(self, session: SudokuSession) -> None:
        sid = session.save.session_id
        # noinspection PyProtectedMember
        session.history._save_as_file(self._session_history_path(sid))
        # noinspection PyProtectedMember
        session.save._save_as_file(self._session_save_path(sid))

    def _generate_starting_field(self) -> Field:
        raise NotImplementedError

    def generate_session(self, name: str, seed: int = 0, difficulty: Difficulty = Difficulty.Medium) -> SudokuSession:
        """Generate session.

        Typically used in 'New Game'.
        """
        history = self._generate_session_history()
        save = SessionSave(name=name, seed=seed, difficulty=difficulty, starting_field=history.boards[0])
        session = SudokuSession(save, history)
        self._save_session_as_file(session)
        return session

    def save_session(self, session: SudokuSession) -> None:
        """Save session for further gaming.

        Typically used in 'Save'.
        """
        kwargs = attrs.asdict(session.save) | {"timestamp": datetime.datetime.now()}
        new_save = SessionSave(**kwargs)
        session.save = new_save
        self._save_session_as_file(session)

    def list_saves(self) -> list[SessionSave]:
        """List saves metadata for further loading.

        Typically used in some kind of 'Continue' menu.
        """
        saves: list[SessionSave] = []
        for save_file in self._saves_directory.glob("*.toml"):
            # noinspection PyProtectedMember
            save = SessionSave._from_file(save_file)
            saves.append(save)
        return saves

    def load_session(self, save: SessionSave) -> SudokuSession:
        """Load session from save.

        Typically used in some kind of 'Continue' menu.
        """
        path = self._session_history_path(save.session_id)
        # noinspection PyProtectedMember
        history = SessionHistory._from_file(path)
        session = SudokuSession(save=save, history=history)
        return session
