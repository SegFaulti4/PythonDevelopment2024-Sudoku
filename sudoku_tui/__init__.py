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


def game(screen: Screen, scenes: list[Scene], scene: Scene, game_model: SudokuModel) -> None:
    """Create game window.

    :param screen: screen object to render Sudoku game in
    :param scenes: list of game scenes
    :param scene: current game scene
    :param game_model: model object that encapsulates game backend
    """
    if not scenes:
        scenes = [
            Scene([StartView(screen, game_model)], -1, name=START_VIEW_SCENE),
            Scene([NewGameView(screen, game_model)], -1, name=NEW_GAME_VIEW_SCENE),
            Scene([LoadGameView(screen, game_model)], -1, name=LOAD_GAME_VIEW_SCENE),
            Scene([], -1, name=GAME_VIEW_SCENE),
        ]
    screen.play(scenes, stop_on_resize=False, start_scene=scene, allow_int=True)
