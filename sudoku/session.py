"""Main game representation."""
from itertools import permutations

from sudoku.types import BoardMask, Num, Pos, SessionData, SessionSave


class SudokuSession:
    """Representation of one game.

    Declares API for initializing the game,
    getting information about board and making step.
    """

    save: SessionSave
    data: SessionData

    def __init__(self, save: SessionSave, data: SessionData) -> None:
        """Game initializer.

        Computes random sudoku board, ready to game.
        """
        self.save = save
        self.data = data
        self.win_flag = False

    @property
    def win(self) -> bool:
        """Check is the game ended.

        If True, no operation on session is valid except win()
        """
        return self.data.boards[self.data.turn] == self.data.full_board

    def undo(self) -> bool:
        """Undo last turn.

        Returns: true if the turn was successful, false otherwise
        """
        if self.win or self.data.turn + len(self.data.boards) == 0:
            return False
        self.data.turn -= 1
        return True

    def redo(self) -> bool:
        """Redo last turn.

        Returns: true if the turn was successful, false otherwise
        """
        if self.win or self.data.turn == -1:
            return False
        self.data.turn += 1
        return True

    def set_num(self, pos: Pos, num: Num) -> bool:
        """Set the point at 'pos' value to 'num'.

        Returns: true if the turn was successful, false otherwise
        """
        if self.win or self.data.initial[pos.x - 1][pos.y - 1]:
            return False
        if self.data.turn != -1:
            del self.data.boards[self.data.turn + 1:]
            self.data.turn = -1
        self.data.boards.append([
            [self.data.boards[self.data.turn][row][col]
             if pos.x - 1 != row or pos.y - 1 != col
             else num
             for col in range(9)] for row in range(9)
        ])
        return True

    def del_num(self, pos: Pos) -> bool:
        """Unset the point at 'pos'.

        Returns: true if the turn was successful, false otherwise
        """
        if self.win or self.data.initial[pos.x - 1][pos.y - 1] \
                or self.data.boards[self.data.turn][pos.x - 1][pos.y - 1] is None:
            return False
        if self.data.turn != -1:
            del self.data.boards[self.data.turn + 1:]
            self.data.turn = -1
        self.data.boards.append([
            [self.data.boards[self.data.turn][row][col]
             if pos.x - 1 != row or pos.y - 1 != col
             else None
             for col in range(9)] for row in range(9)
        ])
        return True

    def get_errors(self) -> BoardMask:
        """Get matrix of errors on board."""
        errors = [[False for __ in range(9)] for __ in range(9)]
        if self.win:
            return errors
        board = self.data.boards[self.data.turn]
        # Check rows and cols
        for i in range(9):
            for j1, j2 in permutations(range(9), 2):
                if board[i][j1] == board[i][j2]:
                    errors[i][j1] = True
                    errors[i][j2] = True
                if board[j1][i] == board[j2][i]:
                    errors[j1][i] = True
                    errors[j2][i] = True
        # Check boxes
        for box in range(9):
            for cell1, cell2 in permutations(range(9), 2):
                r1: int = 3 * (box // 3) + cell1 // 3
                r2: int = 3 * (box // 3) + cell2 // 3
                c1: int = 3 * (box % 3) + cell1 % 3
                c2: int = 3 * (box % 3) + cell2 % 3
                if board[r1][c1] == board[r2][c2]:
                    errors[r1][c1] = True
                    errors[r2][c2] = True
        # Mask errors for unset cells
        for r in range(9):
            for c in range(9):
                if board[r][c] is None:
                    errors[r][c] = False
        return errors

    def get_initials(self) -> list[list[bool]]:
        """Get matrix of initials on board."""
        return self.data.initial.copy()

    def get_board(self) -> list[list[Num | None]]:
        """Get matrix of board values."""
        return self.data.boards[self.data.turn].copy()
