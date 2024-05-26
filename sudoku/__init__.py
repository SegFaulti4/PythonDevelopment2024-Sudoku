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
from math import factorial
from random import choice, getrandbits, sample
from random import seed as randseed
from typing import Any, Callable, Generic, TypeVar, cast

import attrs
import attrs.validators
import cattrs
import cattrs.errors
import numpy as np
import numpy.typing as npt

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
            logger.warning(cattrs.transform_error(exc))
            raise SudokuFileError("Session metadata file corrupted", path=path)
        except Exception as exc:
            logger.warning(exc)
            raise SudokuFileError("Failed to load session metadata file", path=path)

    def _save_as_file(self, path: pathlib.Path) -> None:
        data = attrs.asdict(self)
        try:
            with open(path, "w") as out_f:
                json.dump(data, out_f)
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
        if self.history.turn + len(self.history.boards) == 0:
            return False
        self.history.turn -= 1
        return True

    def redo(self) -> bool:
        """Redo last turn.

        Returns: true if the turn was successful, false otherwise
        """
        if self.history.turn == -1:
            return False
        self.history.turn += 1
        return True

    def set_num(self, pos: Pos, num: Num) -> bool:
        """Set the point at 'pos' value to 'num'.

        Returns: true if the turn was successful, false otherwise
        """
        if self.history.initial[pos.x - 1][pos.y - 1]:
            return False
        if self.history.turn != -1:
            del self.history.boards[self.history.turn + 1:]
            self.history.turn = -1
        self.history.boards.append([
            [self.history.boards[self.history.turn][row][col]
             if pos.x - 1 != row or pos.y - 1 != col
             else num
             for col in range(9)] for row in range(9)
        ])
        return True

    def del_num(self, pos: Pos) -> bool:
        """Unset the point at 'pos'.

        Returns: true if the turn was successful, false otherwise
        """
        if self.history.initial[pos.x - 1][pos.y - 1] \
                or self.history.boards[self.history.turn][pos.x - 1][pos.y - 1] is None:
            return False
        if self.history.turn != -1:
            del self.history.boards[self.history.turn + 1:]
            self.history.turn = -1
        self.history.boards.append([
            [self.history.boards[self.history.turn][row][col]
             if pos.x - 1 != row or pos.y - 1 != col
             else None
             for col in range(9)] for row in range(9)
        ])
        return True

    def get_errors(self) -> BoardMask:
        """Get matrix of errors on board."""
        errors = [[False for __ in range(9)] for __ in range(9)]
        board = self.history.boards[-1]
        # Check rows and cols
        for i in range(9):
            for j1, j2 in permutations(range(9), 2):
                if board[i][j1] == board[i][j2]:
                    errors[i][j1] = True
                    errors[i][j2] = True
                if board[j1][i] == board[j2][i]:
                    errors[j1][i] = True
                    errors[j2][i] = True
        # Check boxes
        for box in range(9):
            for cell1, cell2 in permutations(range(9), 2):
                r1: int = 3 * (box // 3) + cell1 // 3
                r2: int = 3 * (box // 3) + cell2 // 3
                c1: int = 3 * (box % 3) + cell1 % 3
                c2: int = 3 * (box % 3) + cell2 % 3
                if board[r1][c1] == board[r2][c2]:
                    errors[r1][c1] = True
                    errors[r2][c2] = True
        # Mask errors for unset cells
        for r in range(9):
            for c in range(9):
                if board[r][c] is None:
                    errors[r][c] = False
        return errors

    def get_initials(self) -> list[list[bool]]:
        """Get matrix of initials on board."""
        return self.history.initial.copy()

    def get_board(self) -> list[list[Num | None]]:
        """Get matrix of board values."""
        return self.history.boards[self.history.turn].copy()


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
        randseed()

    def _session_history_path(self, session_id: str) -> pathlib.Path:
        return self._saves_directory / (session_id + "-history.json")

    def _session_save_path(self, session_id: str) -> pathlib.Path:
        return self._saves_directory / (session_id + "-save.json")

    def _save_session_as_file(self, session: SudokuSession) -> None:
        sid = session.save.session_id
        # noinspection PyProtectedMember
        session.history._save_as_file(self._session_history_path(sid))
        # noinspection PyProtectedMember
        session.save._save_as_file(self._session_save_path(sid))

    @staticmethod
    def _generate_full_board(seed: str | None) -> FullBoard:
        """Generate full board."""
        def is_correct(field: npt.NDArray[np.int_]) -> bool:
            r = bool(np.unique(field).size == 9 and np.all(np.unique(field) == np.arange(9) + 1)) and \
                all(np.all(np.unique(field[_i, :], return_counts=True)[1] == 1) for _i in range(9)) and \
                all(np.all(np.unique(field[:, _i], return_counts=True)[1] == 1) for _i in range(9))
            return r

        f: npt.NDArray[np.int_] = np.zeros((9, 9), int)

        if seed is not None:
            *seed_boxes, seed_initial = seed.split('-')
            if len(seed_initial) != (81 + 4) // 5 or len(seed_boxes) != 6:
                raise SudokuError("Malformed seed")
            for i, j, box_seed_str in zip((0, 0, 3, 3, 6, 6), (0, 3, 3, 6, 0, 6), seed_boxes):
                if len(box_seed_str) != 4:
                    raise SudokuError("Malformed seed")
                seed_nums = list(map(lambda x: ord(x) - ord('0') if x.isdigit() else ord(x) - ord('a') + 10,
                                     box_seed_str))
                box_seed = sum((32**(3-i) * seed_nums[i]) for i in range(4))
                numbers = list(range(1, 10))
                for k in range(8, 0, -1):
                    f[i + (8-k)//3, j + (8-k)%3] = numbers.pop(box_seed//factorial(k))
                    box_seed = box_seed % factorial(k)
                f[i + 2, j + 2] = numbers[0]
            for i, j in ((0, 6), (3, 0), (6, 3)):
                for k in range(3):
                    for m in range(3):
                        for number in range(1, 10):
                            # Box cannot be deduced if any cell has no numbers available - it'll be 0
                            if number not in f[i + k, :] and number not in f[:, j + m]:
                                f[i + k, j + m] = number
            if not is_correct(f):
                raise SudokuError("Malformed seed")
            return [[Num(f[row][col]) for col in range(9)] for row in range(9)]


        # Creating 1, 5 and 9 boxes
        for i in (0, 3, 6):
            tmp = np.asarray(sample((1, 2, 3, 4, 5, 6, 7, 8, 9), k=9))
            tmp = tmp.reshape((3, 3))
            f[i:i + 3, i:i + 3] = tmp

        # Iteratively trying to generate boxes such that sudoku exists
        res: npt.NDArray[np.int_] = f.copy()
        while not is_correct(res):
            res = f.copy()

            # Trying to generate valid 2, 6 and 7 boxes
            for i, j in ((0, 3), (3, 6), (6, 0)):
                while 0 in res[i:i + 3, j:j + 3]:
                    box_nums = set(range(1, 10))
                    for k in range(3):
                        for m in range(3):
                            avail = list(filter(lambda x: x not in res[i + k, :] and x not in res[:, j + m], box_nums))
                            if len(avail) > 0:
                                res[i + k, j + m] = choice(avail)
                            box_nums.difference_update({res[i + k, j + m]})

            # Trying to deduce 3, 4 and 8 boxes from others
            for i, j in ((0, 6), (3, 0), (6, 3)):
                for k in range(3):
                    for m in range(3):
                        for number in range(1, 10):
                            # Box cannot be deduced if any cell has no numbers available - it'll be 0
                            if number not in res[i + k, :] and number not in res[:, j + m]:
                                res[i + k, j + m] = number

        return [[Num(res[row][col]) for col in range(9)] for row in range(9)]

    @staticmethod
    def _solve(board: Board, difficulty: Difficulty) -> bool:
        marks = [[list(range(1, 10)) for j in range(9)] for i in range(9)]
        def unmark(i: int, j: int) -> None:
            val: int = cast(int, board[i][j])
            for k in [*range(j)] + [*range(j+1, 9)]:
                marks[i][k].remove(val) if val in marks[i][k] else None
            for k in [*range(i)] + [*range(i+1, 9)]:
                marks[k][j].remove(val) if val in marks[k][j] else None
            for k in range(9):
                if val in marks[3*(i//3) + k//3][3*(j//3) + k%3]:
                    marks[3*(i//3) + k//3][3*(j//3) + k%3].remove(val)
            marks[i][j] = []

        for i, j in ((x, y) for x in range(9) for y in range(9)):
            if board[i][j] is not None:
                unmark(i, j)
        changed = True
        while changed:
            changed = False
            # Techniques here

            # 1) Easy technique: naked single
            for i, j in ((x, y) for x in range(9) for y in range(9)):
                if len(marks[i][j]) == 1:
                    board[i][j] = Num(marks[i][j][0])
                    unmark(i, j)
                    changed = True
            # 2) Easy technique: hidden single
            for num in range(1, 10):
                for row in range(9):
                    if sum(num in marks[row][i] for i in range(9)) == 1:
                        for i in range(9):
                            if num in marks[row][i]:
                                board[row][i] = Num(num)
                                unmark(row, i)
                                changed = True
                for col in range(9):
                    if sum(num in marks[i][col] for i in range(9)) == 1:
                        for i in range(9):
                            if num in marks[i][col]:
                                board[i][col] = Num(num)
                                unmark(i, col)
                                changed = True
                for bl in range(9):
                    if sum(num in marks[3*(bl//3) + i//3][3*(bl%3) + i%3] for i in range(9)) == 1:
                        for i in range(9):
                            if num in marks[3*(bl//3) + i//3][3*(bl%3) + i%3]:
                                board[3*(bl//3) + i//3][3*(bl%3) + i%3] = Num(num)
                                unmark(3*(bl//3) + i//3, 3*(bl%3) + i%3)
                                changed = True

        if all(map(all, board)):
            return True
        return False

    @staticmethod
    def _generate_initial_mask(board: FullBoard, seed: str | None, difficulty: Difficulty) -> BoardMask:
        """Generate initial state of board."""
        if seed is not None:
            *_, initial_seed = seed.split('-')
            if len(initial_seed) != 17:
                raise SudokuError("Malformed seed")
            res = []
            for i in range(16):
                val = int(initial_seed[i], base=32)
                res += list(map(lambda x: x == '1', f'{val:05b}'))
            if initial_seed[16] == '0':
                res += [False,]
            elif initial_seed[16] == '1':
                res += [True,]
            else:
                raise SudokuError("Malformed seed")
            return list(res[i:i+9] for i in range(0, 81, 9))

        samples: int = 0
        if difficulty == Difficulty.Easy:
            samples = 40
        elif difficulty == Difficulty.Medium:
            samples = 35
        else:
            samples = 30
        solvable = False
        while not solvable:
            visible: list[tuple[int, int]] = sample(list((x, y) for x in range(9) for y in range(9)), k=samples)
            try_board: Board = [[board[i][j] if (i, j) in visible else None for j in range(9)] for i in range(9)]
            solvable = SudokuServer._solve(try_board, difficulty)
        return [[(i, j) in visible for j in range(9)] for i in range(9)]

    @staticmethod
    def _calc_seed(board: FullBoard, initial: BoardMask) -> str:
        seed = ''
        digits = '0123456789abcdefghijklmnopqrstuvwxyz'
        for i, j in ((0, 0), (0, 3), (3, 3), (3, 6), (6, 0), (6, 6)):
            flat_box = []
            for ii in range(3):
                for jj in range(3):
                    flat_box.append(board[i+ii][j+jj])
            invs = [sum(map(lambda x: flat_box[k] > x, flat_box[k+1:])) for k in range(9)]
            box_seed = 0
            for k in range(9):
                box_seed += invs[k] * factorial(8-k)
            seed_digits = []
            for _ in range(4):
                seed_digits.append(box_seed % 32)
                box_seed //= 32
            seed_digits.reverse()
            seed += ''.join([digits[k] for k in seed_digits])
            seed += '-'
        cur = 0
        cnt = 0
        for i in range(9):
            for j in range(9):
                cur  = 2 * cur + initial[i][j]
                cnt += 1
                if cnt == 5:
                    seed += digits[cur]
                    cnt = 0
                    cur = 0
        if cnt != 0:
            seed += digits[cur]
        return seed


    @staticmethod
    def _generate_session_history(seed: str | None, difficulty: Difficulty) -> SessionHistory:
        full_board = SudokuServer._generate_full_board(seed)
        initial = SudokuServer._generate_initial_mask(full_board, seed=seed, difficulty=difficulty)
        boards: list[Board] = [[[full_board[r][c] if initial[r][c] else None for c in range(9)] for r in range(9)]]
        turn = -1

        history = SessionHistory(full_board, initial, boards, turn)
        return history

    def generate_session(self,
                         name: str,
                         seed: str | None = None,
                         difficulty: Difficulty = Difficulty.Medium) -> SudokuSession:
        """Generate session.

        Typically used in 'New Game'.
        """
        history = SudokuServer._generate_session_history(seed=seed, difficulty=difficulty)
        if seed is None:
            seed = SudokuServer._calc_seed(history.full_board, history.initial)
        save = SessionSave(name=name, session_id=str(uuid.UUID(int=int(getrandbits(128)), version=4)), seed=seed,
                           difficulty=difficulty, starting_field=history.boards[0])
        session = SudokuSession(save, history)
        self._save_session_as_file(session)
        return session

    def save_session(self, session: SudokuSession) -> None:
        """Save session for further gaming.

        Typically used in 'Save'.
        """
        kwargs = attrs.asdict(session.save) | {"timestamp": datetime.datetime.now().isoformat()}
        new_save = SessionSave(**kwargs)
        session.save = new_save
        self._save_session_as_file(session)

    def list_saves(self) -> list[SessionSave]:
        """List saves metadata for further loading.

        Typically used in some kind of 'Continue' menu.
        """
        saves: list[SessionSave] = []
        for save_file in self._saves_directory.glob("*-save.json"):
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
