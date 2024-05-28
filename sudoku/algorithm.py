"""Algorithms of sudoku generation."""
from math import factorial
from random import choice, sample
from typing import cast

import numpy as np
import numpy.typing as npt

from sudoku.exception import SudokuError
from sudoku.types import Board, BoardMask, Difficulty, FullBoard, Num


def generate_full_board(seed: str | None) -> FullBoard:
    """Generate full board."""
    def is_correct(field: npt.NDArray[np.int_]) -> bool:
        r = bool(np.unique(field).size == 9 and np.all(np.unique(field) == np.arange(9) + 1)) and \
            all(np.all(np.unique(field[_i, :], return_counts=True)[1] == 1) for _i in range(9)) and \
            all(np.all(np.unique(field[:, _i], return_counts=True)[1] == 1) for _i in range(9))
        return r

    f: npt.NDArray[np.int_] = np.zeros((9, 9), int)

    if seed is not None:
        *seed_boxes, seed_initial = seed.split('-')
        if len(seed_initial) != (81 + 4) // 5 or len(seed_boxes) != 6:
            raise SudokuError("Malformed seed")
        for i, j, box_seed_str in zip((0, 0, 3, 3, 6, 6), (0, 3, 3, 6, 0, 6), seed_boxes):
            if len(box_seed_str) != 4:
                raise SudokuError("Malformed seed")
            seed_nums = list(map(lambda x: ord(x) - ord('0') if x.isdigit() else ord(x) - ord('a') + 10,
                                    box_seed_str))
            box_seed = sum((32**(3 - i) * seed_nums[i]) for i in range(4))
            numbers = list(range(1, 10))
            for k in range(8, 0, -1):
                f[i + (8 - k) // 3, j + (8 - k) % 3] = numbers.pop(box_seed // factorial(k))
                box_seed = box_seed % factorial(k)
            f[i + 2, j + 2] = numbers[0]
        for i, j in ((0, 6), (3, 0), (6, 3)):
            for k in range(3):
                for m in range(3):
                    for number in range(1, 10):
                        # Box cannot be deduced if any cell has no numbers available - it'll be 0
                        if number not in f[i + k, :] and number not in f[:, j + m]:
                            f[i + k, j + m] = number
        if not is_correct(f):
            raise SudokuError("Malformed seed")
        return [[Num(f[row][col]) for col in range(9)] for row in range(9)]

    # Creating 1, 5 and 9 boxes
    for i in (0, 3, 6):
        tmp = np.asarray(sample((1, 2, 3, 4, 5, 6, 7, 8, 9), k=9))
        tmp = tmp.reshape((3, 3))
        f[i:i + 3, i:i + 3] = tmp

    # Iteratively trying to generate boxes such that sudoku exists
    res: npt.NDArray[np.int_] = f.copy()
    while not is_correct(res):
        res = f.copy()

        # Trying to generate valid 2, 6 and 7 boxes
        for i, j in ((0, 3), (3, 6), (6, 0)):
            while 0 in res[i:i + 3, j:j + 3]:
                box_nums = set(range(1, 10))
                for k in range(3):
                    for m in range(3):
                        avail = list(filter(lambda x: x not in res[i + k, :] and x not in res[:, j + m], box_nums))
                        if len(avail) > 0:
                            res[i + k, j + m] = choice(avail)
                        box_nums.difference_update({res[i + k, j + m]})

        # Trying to deduce 3, 4 and 8 boxes from others
        for i, j in ((0, 6), (3, 0), (6, 3)):
            for k in range(3):
                for m in range(3):
                    for number in range(1, 10):
                        # Box cannot be deduced if any cell has no numbers available - it'll be 0
                        if number not in res[i + k, :] and number not in res[:, j + m]:
                            res[i + k, j + m] = number

    return [[Num(res[row][col]) for col in range(9)] for row in range(9)]


def solve(board: Board, difficulty: Difficulty) -> bool:
    """Sudoku solver.

    Solves board using methods with difficulty not higher than specified.
    Returns if the board solvable .
    """
    marks = [[list(range(1, 10)) for j in range(9)] for i in range(9)]

    def unmark(i: int, j: int) -> None:
        val: int = cast(int, board[i][j])
        for k in [*range(j)] + [*range(j + 1, 9)]:
            marks[i][k].remove(val) if val in marks[i][k] else None
        for k in [*range(i)] + [*range(i + 1, 9)]:
            marks[k][j].remove(val) if val in marks[k][j] else None
        for k in range(9):
            if val in marks[3 * (i // 3) + k // 3][3 * (j // 3) + k % 3]:
                marks[3 * (i // 3) + k // 3][3 * (j // 3) + k % 3].remove(val)
        marks[i][j] = []

    for i, j in ((x, y) for x in range(9) for y in range(9)):
        if board[i][j] is not None:
            unmark(i, j)
    changed = True
    while changed:
        changed = False
        # Techniques here

        # 1) Easy technique: naked single
        for i, j in ((x, y) for x in range(9) for y in range(9)):
            if len(marks[i][j]) == 1:
                board[i][j] = Num(marks[i][j][0])
                unmark(i, j)
                changed = True
        # 2) Easy technique: hidden single
        for num in range(1, 10):
            for row in range(9):
                if sum(num in marks[row][i] for i in range(9)) == 1:
                    for i in range(9):
                        if num in marks[row][i]:
                            board[row][i] = Num(num)
                            unmark(row, i)
                            changed = True
            for col in range(9):
                if sum(num in marks[i][col] for i in range(9)) == 1:
                    for i in range(9):
                        if num in marks[i][col]:
                            board[i][col] = Num(num)
                            unmark(i, col)
                            changed = True
            for bl in range(9):
                if sum(num in marks[3 * (bl // 3) + i // 3][3 * (bl % 3) + i % 3] for i in range(9)) == 1:
                    for i in range(9):
                        if num in marks[3 * (bl // 3) + i // 3][3 * (bl % 3) + i % 3]:
                            board[3 * (bl // 3) + i // 3][3 * (bl % 3) + i % 3] = Num(num)
                            unmark(3 * (bl // 3) + i // 3, 3 * (bl % 3) + i % 3)
                            changed = True

    if all(map(all, board)):
        return True
    return False


def generate_initial_mask(board: FullBoard, seed: str | None, difficulty: Difficulty) -> BoardMask:
    """Generate initial state of board."""
    if seed is not None:
        *_, initial_seed = seed.split('-')
        if len(initial_seed) != 17:
            raise SudokuError("Malformed seed")
        res = []
        for i in range(16):
            val = int(initial_seed[i], base=32)
            res += list(map(lambda x: x == '1', f'{val:05b}'))
        if initial_seed[16] == '0':
            res += [False,]
        elif initial_seed[16] == '1':
            res += [True,]
        else:
            raise SudokuError("Malformed seed")
        return list(res[i:i + 9] for i in range(0, 81, 9))

    samples: int = 0
    if difficulty == Difficulty.Easy:
        samples = 40
    elif difficulty == Difficulty.Medium:
        samples = 35
    else:
        samples = 30
    solvable = False
    while not solvable:
        visible: list[tuple[int, int]] = sample(list((x, y) for x in range(9) for y in range(9)), k=samples)
        try_board: Board = [[board[i][j] if (i, j) in visible else None for j in range(9)] for i in range(9)]
        solvable = solve(try_board, difficulty)
    return [[(i, j) in visible for j in range(9)] for i in range(9)]


def calculate_seed(board: FullBoard, initial: BoardMask) -> str:
    """Calculate seed for given board and initial mask."""
    seed = ''
    digits = '0123456789abcdefghijklmnopqrstuvwxyz'
    for i, j in ((0, 0), (0, 3), (3, 3), (3, 6), (6, 0), (6, 6)):
        flat_box = []
        for ii in range(3):
            for jj in range(3):
                flat_box.append(board[i + ii][j + jj])
        invs = [sum(map(lambda x: flat_box[k] > x, flat_box[k + 1:])) for k in range(9)]
        box_seed = 0
        for k in range(9):
            box_seed += invs[k] * factorial(8 - k)
        seed_digits = []
        for _ in range(4):
            seed_digits.append(box_seed % 32)
            box_seed //= 32
        seed_digits.reverse()
        seed += ''.join([digits[k] for k in seed_digits])
        seed += '-'
    cur = 0
    cnt = 0
    for i in range(9):
        for j in range(9):
            cur = 2 * cur + initial[i][j]
            cnt += 1
            if cnt == 5:
                seed += digits[cur]
                cnt = 0
                cur = 0
    if cnt != 0:
        seed += digits[cur]
    return seed
