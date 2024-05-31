"""Sudoku TUI main menu view."""
from asciimatics.exceptions import NextScene, StopApplication
from asciimatics.screen import Screen
from asciimatics.widgets import Button, Frame, Layout

from sudoku_tui.model import SudokuModel, translate
from sudoku_tui.view.constants import LOAD_GAME_VIEW_SCENE, NEW_GAME_VIEW_SCENE, SUDOKU_THEME


class StartView(Frame):
    """Starting game page."""

    model: SudokuModel

    def __init__(self, screen: Screen, model: SudokuModel) -> None:
        """Create starting game page with existing model."""
        super().__init__(
            screen, screen.height, screen.width,
            title=translate("Main Menu"),
            hover_focus=True,
            can_scroll=False,
            reduce_cpu=True,
        )
        self.palette |= SUDOKU_THEME
        self.model = model

        layout = Layout([100], fill_frame=True)
        self.add_layout(layout)
        layout.add_widget(Button(translate("New Game"),
                                 self._new_game, add_box=False))
        layout.add_widget(Button(translate("Load Game"),
                                 self._load_game, add_box=False))
        layout.add_widget(Button(translate("Exit"),
                                 self._exit, add_box=False))
        self.fix()

    def _new_game(self) -> None:
        raise NextScene(NEW_GAME_VIEW_SCENE)

    def _load_game(self) -> None:
        raise NextScene(LOAD_GAME_VIEW_SCENE)

    @staticmethod
    def _exit() -> None:
        raise StopApplication("Exit the game")
