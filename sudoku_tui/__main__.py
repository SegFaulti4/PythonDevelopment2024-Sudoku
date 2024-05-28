"""Sudoku TUI client entrypoint."""

import sys

from asciimatics.exceptions import ResizeScreenError

import sudoku
from sudoku_tui import Scene, Screen, SudokuModel, game

last_scene = None
server = sudoku.SudokuServer()
model = SudokuModel(server=server)
scenes: list[Scene] = []
while True:
    try:
        Screen.wrapper(game, arguments=[scenes, last_scene, model])
        sys.exit(0)
    except ResizeScreenError as e:
        _last_scene = e.scene
