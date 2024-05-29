"""Sudoku TUI client entrypoint."""

import argparse
import sys

from asciimatics.exceptions import ResizeScreenError

import sudoku
from sudoku_tui import Scene, Screen, SudokuModel, game
from sudoku_tui.model import Locale

parser = argparse.ArgumentParser()
parser.add_argument("--locale", action="store", type=Locale,
                    help="Locale name",
                    required=False, default=Locale.EN)
args = parser.parse_args()

last_scene = None
server = sudoku.SudokuServer()
model = SudokuModel(server=server, locale=args.locale)
scenes: list[Scene] = []
while True:
    try:
        Screen.wrapper(game, arguments=[scenes, last_scene, model])
        sys.exit(0)
    except ResizeScreenError as e:
        _last_scene = e.scene
