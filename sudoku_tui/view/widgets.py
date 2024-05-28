"""Sudoku TUI custom widgets."""
from typing import Any

from asciimatics.screen import Screen
from asciimatics.widgets import Label

# noinspection PyProtectedMember
from asciimatics.widgets.utilities import _split_text

from sudoku_tui.model import ALL_NUMS, sudoku

CAKE = """

' ' ' '
| | | |
@@@@@@@@@@@@@@@
{~@~@~@~@~@~@~}
@@@@@@@@@@@@@@@@@@@@@@@
{~@ CONGRATULATIONS @~}
{   you're awesome!   }
@@@@@@@@@@@@@@@@@@@@@@@
_)_______(_
|===========|"""


class SudokuBoardWidget(Label):
    """Custom widget for board output."""

    board: sudoku.Board
    initial: sudoku.BoardMask
    errors: sudoku.BoardMask

    UNSET_CELL = "*"

    def __init__(self, *args: Any, board: sudoku.Board, initial: sudoku.BoardMask,
                 errors: sudoku.BoardMask, **kwargs: Any) -> None:
        """Create board widget from session data."""
        super().__init__(*args, **kwargs)
        self.board = board
        self.initial = initial
        self.errors = errors
        self._text = self._board_repr(self.board)

    def update(self, frame_no: int = 0) -> None:
        """Render board."""
        self._text = self._board_repr(self.board)

        cell_idx = 0
        offset = max(self._w - 24, 0) // 2
        for i, line in enumerate(_split_text(
                self._text, self._w, self._h,
                self._frame.canvas.unicode_aware)):
            for j, char in reversed(list(enumerate(line))):
                if j == 0:
                    text = f"{char:>{offset}}"
                    x = self._x
                else:
                    text = char
                    x = self._x + offset + j - 1

                fg, at, bg = Screen.COLOUR_WHITE, Screen.A_NORMAL, Screen.COLOUR_BLACK
                if char == self.UNSET_CELL or char in ALL_NUMS:
                    if self.errors[cell_idx // 9][8 - (cell_idx % 9)]:
                        fg = Screen.COLOUR_RED
                    elif self.initial[cell_idx // 9][8 - (cell_idx % 9)]:
                        fg = Screen.COLOUR_DEFAULT
                    cell_idx += 1

                self._frame.canvas.paint(text, x, self._y + i, fg, at, bg)

    @staticmethod
    def _num_repr(num: sudoku.Num | None) -> str:
        if num is None:
            return SudokuBoardWidget.UNSET_CELL
        return f"{num.value}"

    @staticmethod
    def _row_repr(row: list[sudoku.Num | None]) -> str:
        return "| " + " | ".join(
            " ".join(SudokuBoardWidget._num_repr(num) for num in row[start:end])
            for start, end in [(0, 3), (3, 6), (6, 9)]
        ) + " |"

    @staticmethod
    def _board_repr(board: sudoku.Board) -> str:
        return f"\n+{'=' * 23}+\n" + f"\n|{'=' * 23}|\n".join(
            "\n".join(SudokuBoardWidget._row_repr(row) for row in board[start:end])
            for start, end in [(0, 3), (3, 6), (6, 9)]
        ) + f"\n+{'=' * 23}+"


class ColoredLabel(Label):
    """Custom widget for label with set colors."""

    foreground: int
    attribute: int
    background: int

    def __init__(self, *args: Any,
                 foreground: int = Screen.COLOUR_WHITE,
                 attribute: int = Screen.A_NORMAL,
                 background: int = Screen.COLOUR_BLACK,
                 **kwargs: Any) -> None:
        """Create board widget from session data."""
        super().__init__(*args, **kwargs)
        self.foreground = foreground
        self.attribute = attribute
        self.background = background

    def update(self, frame_no: int) -> None:
        """Render label."""
        for i, text in enumerate(
                _split_text(self._text, self._w, self._h, self._frame.canvas.unicode_aware)):
            self._frame.canvas.paint(
                f"{text:{self._align}{self._w}}",
                self._x, self._y + i,
                self.foreground, self.attribute, self.background
            )
