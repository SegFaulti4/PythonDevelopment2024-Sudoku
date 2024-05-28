"""Sudoku TUI client package."""
from asciimatics.scene import Scene
from asciimatics.screen import Screen

from sudoku_tui.model import SudokuModel
from sudoku_tui.view import (
    GAME_VIEW_SCENE,
    LOAD_GAME_VIEW_SCENE,
    NEW_GAME_VIEW_SCENE,
    START_VIEW_SCENE,
    LoadGameView,
    NewGameView,
    StartView,
)


def game(screen: Screen, _scenes: list[Scene], _scene: Scene, _model: SudokuModel) -> None:
    """Create game window."""
    if not _scenes:
        _scenes = [
            Scene([StartView(screen, _model)], -1, name=START_VIEW_SCENE),
            Scene([NewGameView(screen, _model)], -1, name=NEW_GAME_VIEW_SCENE),
            Scene([LoadGameView(screen, _model)], -1, name=LOAD_GAME_VIEW_SCENE),
            Scene([], -1, name=GAME_VIEW_SCENE),
        ]
    screen.play(_scenes, stop_on_resize=False, start_scene=_scene, allow_int=True)
