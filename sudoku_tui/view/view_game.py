"""Sudoku TUI game view."""
import typing

from asciimatics.exceptions import NextScene
from asciimatics.screen import Screen
from asciimatics.widgets import Button, Divider, Frame, Layout, Text

from sudoku_tui.model import ALL_NUMS, SudokuModel, sudoku
from sudoku_tui.view.constants import START_VIEW_SCENE
from sudoku_tui.view.widgets import CAKE, ColoredLabel, SudokuBoardWidget


class GameView(Frame):
    """Game session page."""

    __scene_name__ = "Sudoku"
    model: SudokuModel

    _board_widget: SudokuBoardWidget
    _tips_widget: ColoredLabel
    _prompt_widget: Text
    _log_widget: ColoredLabel
    _status_widget: ColoredLabel
    _save_button: Button
    _exit_button: Button

    _old_prompt_value: str = ""

    # Command prompt related constants
    _UNDO_COMMAND = "u"
    _REDO_COMMAND = "r"
    _WIN_COMMAND = "win"
    _DEL_NUM_CHAR = SudokuBoardWidget.UNSET_CELL
    _PROMPT_MAX_LEN = max(
        len(_UNDO_COMMAND),
        len(_REDO_COMMAND),
        len(_WIN_COMMAND),
        3  # set- and del- num commands
    )
    _PROMPT_TIPS = [
        "Command examples:",
        ">> 123 # set [1][2] cell (1-based index) as 3",
        f">> 12{_DEL_NUM_CHAR} # clear [1][2] cell",
        f">> {_UNDO_COMMAND} # undo last change",
        f">> {_REDO_COMMAND} # redo last change",
    ]

    def __init__(self, screen: Screen, model: SudokuModel) -> None:
        """Create game session page with existing model."""
        super().__init__(
            screen, screen.height, screen.width,
            title=self.__scene_name__,
            hover_focus=True,
            can_scroll=False,
            reduce_cpu=True,
        )
        self.model = model
        assert (self.model.session is not None)

        board_layout = Layout([1, 1, 1], fill_frame=True)
        self._tips_widget = ColoredLabel(
            "\n" + "\n".join(self._PROMPT_TIPS), height=2 + len(self._PROMPT_TIPS),
            foreground=Screen.COLOUR_DEFAULT,
        )
        self._board_widget = SudokuBoardWidget(
            "", height=14,
            board=self.model.session.get_board(),
            initial=self.model.session.get_initials(),
            errors=self.model.session.get_errors()
        )
        self._prompt_widget = Text(
            label=">>",
            max_length=self._PROMPT_MAX_LEN,
            on_change=self._prompt_handler,
        )
        self._log_widget = ColoredLabel(
            "", height=2,
            foreground=Screen.COLOUR_RED,
        )
        self._status_widget = ColoredLabel(
            "", height=14, align="^",
        )

        self.add_layout(board_layout)
        board_layout.add_widget(self._tips_widget, 0)
        board_layout.add_widget(self._prompt_widget, 0)
        board_layout.add_widget(self._log_widget, 0)
        board_layout.add_widget(self._board_widget, 1)
        board_layout.add_widget(self._status_widget, 2)

        div_layout = Layout([100])
        self.add_layout(div_layout)
        div_layout.add_widget(Divider())

        bottom_layout = Layout([1, 1])
        self._save_button = Button("Save", self._save_button_handler)
        self._exit_button = Button("Leave", self._exit_button_handler)
        self.add_layout(bottom_layout)
        bottom_layout.add_widget(self._save_button, 0)
        bottom_layout.add_widget(self._exit_button, 1)
        self.fix()
        self._update_board()

    def _prompt_handler(self) -> None:
        assert self.model.session is not None
        old = self._old_prompt_value
        new = self._prompt_widget.value
        self._old_prompt_value = new

        if len(old) > len(new):  # only happens if Backspace was pressed
            return
        if not (
            self._UNDO_COMMAND.startswith(new)
            or self._REDO_COMMAND.startswith(new)
            or self._WIN_COMMAND.startswith(new)
            or (all(c in ALL_NUMS for c in new[:2])
                and len(new) <= 3
                and (len(new) < 3 or new[2] in ALL_NUMS + self._DEL_NUM_CHAR))
        ):
            self._clear_prompt()
            return

        if new == self._WIN_COMMAND:
            self.model.session.data.boards[-1] = typing.cast(sudoku.Board, self.model.session.data.full_board)
            self._clear_prompt()
            self._update_board()
        elif new == self._UNDO_COMMAND:
            self.model.session.undo()
            self._clear_prompt()
            self._update_board()
        elif new == self._REDO_COMMAND:
            self.model.session.redo()
            self._clear_prompt()
            self._update_board()
        elif len(new) == 3 and all(c in ALL_NUMS for c in new[:3]):
            self._handle_set_num_prompt(new)
        elif len(new) == 3 and all(c in ALL_NUMS for c in new[:2]) \
                and new[2] == self._DEL_NUM_CHAR:
            self._handle_del_num_prompt(new)

    def _handle_set_num_prompt(self, new: str) -> None:
        try:
            x = sudoku.Num(int(new[0]))
            y = sudoku.Num(int(new[1]))
            num = sudoku.Num(int(new[2]))
        except Exception as exc:
            self._log_widget.text = f"\nWrong set num command:\n{exc.args[0]}"
        else:
            assert self.model.session is not None
            self.model.session.set_num(sudoku.Pos(x, y), num)
            self._update_board()
        finally:
            self._clear_prompt()

    def _handle_del_num_prompt(self, new: str) -> None:
        try:
            x = sudoku.Num(int(new[0]))
            y = sudoku.Num(int(new[1]))
        except Exception as exc:
            self._log_widget.text = f"\nWrong del num command:\n{exc.args[0]}"
        else:
            assert self.model.session is not None
            self.model.session.del_num(sudoku.Pos(x, y))
            self._update_board()
        finally:
            self._clear_prompt()

    def _save_button_handler(self) -> None:
        assert self.model.session is not None
        self.model.server.save_session(self.model.session)

    def _exit_button_handler(self) -> None:
        assert self.model.session is not None
        # self.model.server.save_session(self.model.session)
        raise NextScene(START_VIEW_SCENE)

    def _clear_prompt(self) -> None:
        """Clear non-empty prompt value after new symbol was input.

        Should only be used by prompt handler.
        """
        self._old_prompt_value = ""
        self._prompt_widget._value = ""
        self._prompt_widget._start_column = 0
        # HACK: _column must be zero as a result, but it is always incremented by 1
        #  after new symbol was input, so we need to compensate for that
        self._prompt_widget._column = -1

    def _update_board(self) -> None:
        assert self.model.session is not None
        self._board_widget.board = self.model.session.get_board()
        self._board_widget.errors = self.model.session.get_errors()
        self._board_widget.update()
        if self.model.session.win:
            self._prompt_widget.disabled = True
            self._status_widget.text = CAKE
            self._log_widget.text = "\nCommand prompt is disabled"
