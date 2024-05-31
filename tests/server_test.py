import unittest
from itertools import permutations, product

import sudoku
from sudoku import Num, Pos


class TestSudokuServer(unittest.TestCase):

    def setUp(self) -> None:
        self.server = sudoku.SudokuServer()

    def tearDown(self):
        for save in self.server.list_saves():
            self.server.delete_save(save)

    def test_save_load(self) -> None:
        s1 = self.server.generate_session("test_save")
        self.server.save_session(s1)
        s2 = self.server.load_session(self.server.list_saves()[0])
        self.assertEqual(s1.data, s2.data)
        self.assertEqual(s1.save, s2.save)

    def test_gen_fullboard(self) -> None:
        session = self.server.generate_session("test_gen_fullboard")
        board = session.data.full_board
        # Check rows and cols
        for i in range(9):
            for j1, j2 in permutations(range(9), 2):
                self.assertNotEqual(board[i][j1], board[i][j2])
                self.assertNotEqual(board[j1][i], board[j2][i])
        # Check boxes
        for box in range(9):
            for cell1, cell2 in permutations(range(9), 2):
                r1: int = 3 * (box // 3) + cell1 // 3
                r2: int = 3 * (box // 3) + cell2 // 3
                c1: int = 3 * (box % 3) + cell1 % 3
                c2: int = 3 * (box % 3) + cell2 % 3
                self.assertNotEqual(board[r1][c1], board[r2][c2])

    def test_undo_redo(self) -> None:
        session = self.server.generate_session("test_undo_redo")
        old_board = session.get_board()
        i, j = 1, 1
        while not session.set_num(Pos(Num(i), Num(j)), Num(1)):
            j += 1
            if j == 10:
                i += 1
                j = 1
        new_board = session.get_board()
        session.undo()
        self.assertEqual(old_board, session.get_board())
        session.redo()
        self.assertEqual(new_board, session.get_board())

    def test_set_num(self) -> None:
        session = self.server.generate_session("test_undo_redo")
        for i, j in product(range(1, 10), repeat=2):
            self.assertNotEqual(session.set_num(Pos(Num(i), Num(j)), Num(1)), session.data.initial[i - 1][j - 1])
        self.assertTrue(all(row) for row in session.get_board())
        for i, j in product(range(1, 10), repeat=2):
            self.assertNotEqual(session.del_num(Pos(Num(i), Num(j))), session.data.initial[i - 1][j - 1])
        self.assertTrue(all((session.get_board()[i][j] is not None) == session.data.initial[i][j]
                            for i in range(9) for j in range(9)))

    def test_seed(self) -> None:
        s1 = self.server.generate_session("test_seed_1")
        s2 = self.server.generate_session("test_seed_2", seed=s1.save.seed)
        self.assertEqual(s1.save.seed, s2.save.seed)
        self.assertEqual(s1.data, s2.data)


if __name__ == "__main__":
    unittest.main()
