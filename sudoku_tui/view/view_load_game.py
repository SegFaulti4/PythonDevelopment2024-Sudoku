"""Sudoku TUI load game view."""
from asciimatics.exceptions import NextScene
from asciimatics.screen import Screen
from asciimatics.widgets import Button, Divider, Frame, Layout, MultiColumnListBox, Widget

from sudoku_tui.model import SudokuModel, sudoku
from sudoku_tui.view.constants import GAME_VIEW_SCENE, START_VIEW_SCENE
from sudoku_tui.view.view_game import GameView


class LoadGameView(Frame):
    """Game saves page."""

    __scene_name__ = "Load Game"
    model: SudokuModel

    _list_widget: MultiColumnListBox
    _load_button: Button
    _delete_button: Button
    _cancel_button: Button

    def __init__(self, screen: Screen, model: SudokuModel) -> None:
        """Create game saves page with existing model."""
        super().__init__(
            screen, screen.height, screen.width,
            title=self.__scene_name__,
            hover_focus=True,
            can_scroll=True,
            reduce_cpu=True,
        )
        self.model = model
        self.model.update_saves()
        self._list_widget = MultiColumnListBox(
            Widget.FILL_FRAME,
            ["<50%", ">50%"],
            self._saves_options(self.model.saves),
            add_scroll_bar=True,
            on_change=self._on_pick,
            on_select=self._load_button_handler,
        )
        self._load_button = Button("Load", self._load_button_handler)
        self._delete_button = Button("Delete", self._delete_button_handler)
        self._cancel_button = Button("Cancel", self._cancel_button_handler)

        main_layout = Layout([100], fill_frame=True)
        self.add_layout(main_layout)
        main_layout.add_widget(self._list_widget)
        main_layout.add_widget(Divider())

        bottom_layout = Layout([1, 1, 1])
        self.add_layout(bottom_layout)
        bottom_layout.add_widget(self._load_button, 0)
        bottom_layout.add_widget(self._delete_button, 1)
        bottom_layout.add_widget(self._cancel_button, 2)
        self.fix()
        self._on_pick()

    def reset(self) -> None:
        """Reload saves page."""
        super().reset()
        self.model.update_saves()
        self._reset()

    def _on_pick(self) -> None:
        self._load_button.disabled = self._list_widget.value is None
        self._delete_button.disabled = self._list_widget.value is None

    def _reset(self, new_value: sudoku.SessionSave | None = None) -> None:
        self._list_widget.options = self._saves_options(self.model.saves)
        self._list_widget.value = new_value

    @staticmethod
    def _saves_options(saves: list[sudoku.SessionSave]) -> list[tuple[list[str], sudoku.SessionSave]]:
        return [
            ([f'"{save.name}"  ({save.difficulty})  [{save.timestamp}]', f"{{{save.seed}}}"],
             save)
            for save in saves
        ]

    def _load_button_handler(self) -> None:
        save = self._list_widget.value
        session = self.model.server.load_session(save)
        self.model.session = session

        screen: Screen = self.screen
        # noinspection PyProtectedMember
        # NOTE: last scene is always reserved for session view
        screen._scenes[-1].add_effect(GameView(self.screen, self.model))
        raise NextScene(GAME_VIEW_SCENE)

    def _delete_button_handler(self) -> None:
        save = self._list_widget.value
        self.model.server.delete_save(save)
        self.model.saves.remove(save)
        self._reset()

    def _cancel_button_handler(self) -> None:
        raise NextScene(START_VIEW_SCENE)
