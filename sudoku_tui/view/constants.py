"""Sudoku TUI view constants."""
from asciimatics.screen import Screen

START_VIEW_SCENE = "Main Menu"
NEW_GAME_VIEW_SCENE = "New Game"
LOAD_GAME_VIEW_SCENE = "Load Game"
GAME_VIEW_SCENE = "Sudoku"

SUDOKU_THEME = {
    "background": (Screen.COLOUR_WHITE, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "shadow": (Screen.COLOUR_BLACK, None, Screen.COLOUR_BLACK),
    "disabled": (Screen.COLOUR_BLACK, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "invalid": (Screen.COLOUR_BLACK, Screen.A_NORMAL, Screen.COLOUR_RED),
    "label": (Screen.COLOUR_DEFAULT, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "borders": (Screen.COLOUR_DEFAULT, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "scroll": (Screen.COLOUR_DEFAULT, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "title": (Screen.COLOUR_WHITE, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "edit_text": (Screen.COLOUR_DEFAULT, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "focus_edit_text": (Screen.COLOUR_WHITE, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "readonly": (Screen.COLOUR_BLACK, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "focus_readonly": (Screen.COLOUR_DEFAULT, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "button": (Screen.COLOUR_WHITE, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "focus_button": (Screen.COLOUR_WHITE, Screen.A_REVERSE, Screen.COLOUR_BLACK),
    "control": (Screen.COLOUR_CYAN, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "selected_control": (Screen.COLOUR_YELLOW, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "focus_control": (Screen.COLOUR_CYAN, Screen.A_REVERSE, Screen.COLOUR_BLACK),
    "selected_focus_control": (Screen.COLOUR_YELLOW, Screen.A_REVERSE, Screen.COLOUR_BLACK),
    "field": (Screen.COLOUR_DEFAULT, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "selected_field": (Screen.COLOUR_DEFAULT, Screen.A_REVERSE, Screen.COLOUR_BLACK),
    "focus_field": (Screen.COLOUR_WHITE, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    "selected_focus_field": (Screen.COLOUR_WHITE, Screen.A_REVERSE, Screen.COLOUR_BLACK),
}
