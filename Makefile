python_bin := python3.10
venv_bin := venv/bin
modules := clients sudoku

mypy_targets := $(addprefix mypy/,$(modules))
ruff_targets := $(addprefix ruff/,$(modules))
ruff_fix_targets := $(addprefix ruff/fix/,$(modules))

all: venv pyproject.toml _check_poetry
	. $(venv_bin)/activate && poetry install

mypy: $(mypy_targets)

ruff: $(ruff_targets)

ruff-fix: $(ruff_fix_targets)

$(mypy_targets): mypy/%: venv
	$(venv_bin)/mypy $*

$(ruff_targets): ruff/%: venv
	$(venv_bin)/ruff check $*

$(ruff_fix_targets): ruff/fix/%: venv
	$(venv_bin)/ruff check --fix $*

venv:
	$(python_bin) -m venv venv

_check_poetry:
	poetry --version


.SILENT: _check_poetry
.PHONY: all mypy ruff ruff-fix _check_poetry $(mypy_targets) $(ruff_targets) $(ruff_fix_targets)
