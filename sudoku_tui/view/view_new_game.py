"""Sudoku TUI new game view."""
from asciimatics.exceptions import NextScene
from asciimatics.screen import Screen
from asciimatics.widgets import Button, Divider, Frame, Layout, ListBox, Text

from sudoku_tui.model import SudokuModel, sudoku, translate
from sudoku_tui.view.constants import GAME_VIEW_SCENE, START_VIEW_SCENE
from sudoku_tui.view.view_game import GameView


class NewGameView(Frame):
    """New game inputs page."""

    __scene_name__ = "New Game"
    model: SudokuModel

    _name_widget: Text
    _difficulty_widget: ListBox
    _seed_widget: Text

    def __init__(self, screen: Screen, model: SudokuModel) -> None:
        """Create new game page with existing model."""
        super().__init__(
            screen, screen.height, screen.width,
            title=translate("New Game"),
            hover_focus=True,
            can_scroll=False,
            reduce_cpu=True,
        )
        self.model = model

        difficulties: list[tuple[str, str]] = [(d, d) for d in sudoku.Difficulty]
        self._name_widget = Text(translate("Name:"),
                                 "name")
        self._difficulty_widget = ListBox(
            len(difficulties),
            difficulties,
            name="difficulty",
            label=translate("Difficulty:"),
        )
        self._seed_widget = Text(translate("Seed:"),
                                 "seed")
        main_layout = Layout([100], fill_frame=True)

        self.add_layout(main_layout)
        main_layout.add_widget(self._name_widget)
        main_layout.add_widget(self._difficulty_widget)
        main_layout.add_widget(self._seed_widget)

        div_layout = Layout([100])
        self.add_layout(div_layout)
        div_layout.add_widget(Divider())

        buttons_layout = Layout([1, 1])
        self.add_layout(buttons_layout)
        buttons_layout.add_widget(Button(translate("OK"),
                                         self._ok_button_handler), 0)
        buttons_layout.add_widget(Button(translate("Cancel"),
                                         self._cancel_button_handler), 1)
        self.fix()

    def reset(self) -> None:
        """Clear new game inputs."""
        super().reset()
        self._name_widget.value = ""
        self._difficulty_widget.value = sudoku.Difficulty.Medium
        self._seed_widget.value = ""

    def _ok_button_handler(self) -> None:
        session = self.model.server.generate_session(
            name=self._name_widget.value,
            difficulty=self._difficulty_widget.value,
            seed=self._seed_widget.value if self._seed_widget.value else None,
        )
        self.model.session = session

        screen: Screen = self.screen
        # noinspection PyProtectedMember
        # NOTE: last scene is always reserved for session view
        screen._scenes[-1].add_effect(GameView(self.screen, self.model))
        raise NextScene(GAME_VIEW_SCENE)

    @staticmethod
    def _cancel_button_handler() -> None:
        raise NextScene(START_VIEW_SCENE)
