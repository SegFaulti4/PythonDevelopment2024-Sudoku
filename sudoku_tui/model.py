"""Sudoku TUI client model."""

import enum
import gettext
import os.path

import sudoku


class Locale(str, enum.Enum):
    """Set of supported locales."""

    RU = "ru_RU.UTF-8"
    EN = "en_US.UTF-8"


ALL_NUMS = "".join(str(n.value) for n in sudoku.Num)

_po_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'po'))
LOCALES = {
    Locale.RU: gettext.translation("sudoku_tui", _po_dir, ["ru"], fallback=False),
    Locale.EN: gettext.NullTranslations(),
}


def translate(locale: Locale, text: str) -> str:
    """Translate text according to current locale."""
    return LOCALES[locale].gettext(text)


class SudokuModel:
    """'Model part' of Model-View client architecture."""

    server: sudoku.SudokuServer
    locale: Locale
    session: sudoku.SudokuSession | None = None
    saves: list[sudoku.SessionSave] = []

    def __init__(self, server: sudoku.SudokuServer, locale: Locale = Locale.EN) -> None:
        """Create model with existing server."""
        self.server = server
        self.locale = locale

    def update_saves(self) -> None:
        """Update saves list by server."""
        self.saves = self.server.list_saves()
