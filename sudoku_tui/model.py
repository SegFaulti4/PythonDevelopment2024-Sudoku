"""Sudoku TUI client model."""

import gettext
import os.path

import sudoku

ALL_NUMS = "".join(str(n.value) for n in sudoku.Num)

_po_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'po'))

CUR_LOCALE = gettext.translation("sudoku_tui", _po_dir, fallback=True)


def translate(text: str) -> str:
    """Translate text according to current locale.

    :param locale: locale to translate to
    :param text: text to translate
    """
    return CUR_LOCALE.gettext(text)


class SudokuModel:
    """'Model part' of Model-View client architecture."""

    server: sudoku.SudokuServer
    session: sudoku.SudokuSession | None = None
    saves: list[sudoku.SessionSave] = []

    def __init__(self, server: sudoku.SudokuServer) -> None:
        """Create model with existing server."""
        self.server = server

    def update_saves(self) -> None:
        """Update saves list by server."""
        self.saves = self.server.list_saves()
