"""Submodule for main logic implementation(servers).

API for client-server interaction provided.
"""

import datetime
import pathlib
import uuid
from random import getrandbits
from random import seed as randseed

import attrs

import sudoku.algorithm as alg
from sudoku.session import SudokuSession
from sudoku.types import Board, Difficulty, SessionData, SessionSave


class SudokuServer:
    """Representation of sudoku server.

    Implements all server-side logic except game managing itself.
    """

    _config_directory: pathlib.Path = pathlib.Path.home() / ".config" / "cmc_sudoku_2024"
    _saves_directory: pathlib.Path = _config_directory / "saves"

    def __init__(self) -> None:
        """Server initializer."""
        self._config_directory.mkdir(parents=True, exist_ok=True)
        self._saves_directory.mkdir(parents=True, exist_ok=True)
        randseed()

    def _session_data_path(self, session_id: str) -> pathlib.Path:
        return self._saves_directory / (session_id + "-data.json")

    def _session_save_path(self, session_id: str) -> pathlib.Path:
        return self._saves_directory / (session_id + "-save.json")

    def _save_session_as_file(self, session: SudokuSession) -> None:
        sid = session.save.session_id
        # noinspection PyProtectedMember
        session.data._save_as_file(self._session_data_path(sid))
        # noinspection PyProtectedMember
        session.save._save_as_file(self._session_save_path(sid))

    def generate_session(self,
                         name: str,
                         seed: str | None = None,
                         difficulty: Difficulty = Difficulty.Medium) -> SudokuSession:
        """Generate session.

        Typically used in 'New Game'.

        :param name: session name
        :param seed: RNG seed
        :param difficulty: difficulty of the session
        """
        data = SudokuServer._generate_session_data(seed=seed, difficulty=difficulty)
        if seed is None:
            seed = alg.calculate_seed(data.full_board, data.initial)
        save = SessionSave(name=name, session_id=str(uuid.UUID(int=int(getrandbits(128)), version=4)), seed=seed,
                           difficulty=difficulty, starting_field=data.boards[0])
        session = SudokuSession(save, data)
        self._save_session_as_file(session)
        return session

    def save_session(self, session: SudokuSession) -> None:
        """Save session for further gaming.

        Typically used in 'Save'.

        :param session: session to save
        """
        kwargs = attrs.asdict(session.save) | {"timestamp": datetime.datetime.now().isoformat()}
        new_save = SessionSave(**kwargs)
        session.save = new_save
        self._save_session_as_file(session)

    def list_saves(self) -> list[SessionSave]:
        """List saves metadata for further loading.

        Typically used in some kind of 'Continue' menu.
        """
        saves: list[SessionSave] = []
        for save_file in self._saves_directory.glob("*-save.json"):
            # noinspection PyProtectedMember
            save = SessionSave._from_file(save_file)
            saves.append(save)
        return saves

    def delete_save(self, save: SessionSave) -> None:
        """Delete save.

        Completely deletes save. It cannot be restored in any way.

        :param save: save to delete
        """
        self._session_data_path(save.session_id).unlink()
        self._session_save_path(save.session_id).unlink()

    def load_session(self, save: SessionSave) -> SudokuSession:
        """Load session from save.

        Typically used in some kind of 'Continue' menu.

        :param save: save to load
        """
        path = self._session_data_path(save.session_id)
        # noinspection PyProtectedMember
        data = SessionData._from_file(path)
        session = SudokuSession(save=save, data=data)
        return session

    @staticmethod
    def _generate_session_data(seed: str | None, difficulty: Difficulty) -> SessionData:
        full_board = alg.generate_full_board(seed)
        initial = alg.generate_initial_mask(full_board, seed=seed, difficulty=difficulty)
        boards: list[Board] = [[[full_board[r][c] if initial[r][c] else None for c in range(9)] for r in range(9)]]
        turn = -1

        data = SessionData(full_board, initial, boards, turn)
        return data
