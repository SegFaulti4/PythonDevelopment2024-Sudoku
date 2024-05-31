"""Sudoku TUI client entrypoint."""

import sys

from asciimatics.exceptions import ResizeScreenError

import sudoku
from sudoku_tui import Scene, Screen, SudokuModel, game

_last_scene = None
_server = sudoku.SudokuServer()
_model = SudokuModel(server=_server)
_scenes: list[Scene] = []
while True:
    try:
        Screen.wrapper(game, arguments=[_scenes, _last_scene, _model])
        sys.exit(0)
    except ResizeScreenError as e:
        _last_scene = e.scene
