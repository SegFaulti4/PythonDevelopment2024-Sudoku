# PythonDevelopment2024-Sudoku

Required python version >=3.10.

To install project from source [poetry](https://python-poetry.org/) is required.

## Play game

After installing Sudoku from source or from release package the game is available as python module:
```shell
python3 -m sudoku_tui
```

## Make commands

Install project and its dependencies from source:
```shell
make all
```

> The rest of commands require `make all` to be run first

Build wheel and sdist of the project (including locale and docs):
```shell
make build
```

Run all tests and codestyle checks:
```shell
make check
```

Generate html docs:
```shell
make docs
```

Generate russian locale:
```shell
make locale
```

Run tests:
```shell
make tests
```

Run mypy checks:
```shell
make mypy
```

Run ruff checks:
```shell
make ruff
```

Run ruff auto-fixes:
```shell
make ruff-fix
```
