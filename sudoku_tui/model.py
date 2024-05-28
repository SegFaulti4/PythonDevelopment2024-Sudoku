"""Sudoku TUI client model."""

import sudoku

ALL_NUMS = "".join(str(n.value) for n in sudoku.Num)


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
