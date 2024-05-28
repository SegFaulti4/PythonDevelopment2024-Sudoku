"""Types for Sudoku backend."""
from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import pathlib
from typing import Any, Callable, Generic, TypeVar

import attrs
import attrs.validators
import cattrs

from sudoku.exception import LOG, SudokuFileError


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


class Difficulty(enum.IntEnum):
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
    seed: str
    difficulty: Difficulty
    starting_field: Board = attrs.field(validator=[_pv(_pred_9x9)])
    session_id: str
    timestamp: str = datetime.datetime.now().isoformat()

    @staticmethod
    def _from_file(path: pathlib.Path) -> SessionSave:
        try:
            with open(path, "r") as in_f:
                data = json.load(in_f)
            save = cattrs.structure(data, SessionSave)
            return save
        except cattrs.errors.ClassValidationError as exc:
            LOG.warning(cattrs.transform_error(exc))
            raise SudokuFileError("Session metadata file corrupted", path=path)
        except Exception as exc:
            LOG.warning(exc)
            raise SudokuFileError("Failed to load session metadata file", path=path)

    def _save_as_file(self, path: pathlib.Path) -> None:
        data = attrs.asdict(self)
        try:
            with open(path, "w") as out_f:
                json.dump(data, out_f)
        except Exception as exc:
            LOG.warning(exc)
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
            LOG.warning(cattrs.transform_error(exc))
            raise SudokuFileError("Save field file corrupted", path=path)
        except Exception as exc:
            LOG.warning(exc)
            raise SudokuFileError("Failed to read field file", path=path)

    def _save_as_file(self, path: pathlib.Path) -> None:
        data = attrs.asdict(self)
        try:
            with open(path, "w") as out_f:
                json.dump(data, out_f)
        except Exception as exc:
            LOG.warning(exc)
            raise SudokuFileError("Failed to save session history file", path=path)
