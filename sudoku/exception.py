"""Sudoku backend exceptions and logger."""

import logging
import pathlib
from typing import Any

LOG = logging.getLogger()
LOG.setLevel(logging.DEBUG)


class SudokuError(Exception):
    """Base exception class for sudoku backend."""


class SudokuFileError(SudokuError):
    """Files related exception class."""

    path: pathlib.Path

    def __init__(self, *args: Any, path: pathlib.Path):
        """Create file error with path."""
        super().__init__(*args)
        self.path = path
