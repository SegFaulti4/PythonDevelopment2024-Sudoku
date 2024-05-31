python_bin := python3.10
tui_module := sudoku_tui
modules := $(tui_module) sudoku tests

po_base := po
po_locale := ru_RU.UTF-8
po_dir := $(po_base)/$(po_locale)/LC_MESSAGES

venv_target := venv
pot_file := tui.pot
po_file := $(po_dir)/$(tui_module).po
mo_file := $(po_dir)/$(tui_module).mo
mypy_targets := $(addprefix mypy/,$(modules))
ruff_targets := $(addprefix ruff/,$(modules))
ruff_fix_targets := $(addprefix ruff/fix/,$(modules))

venv_bin := $(venv_target)/bin

all: venv pyproject.toml _check_poetry
	. $(venv_bin)/activate && poetry install

build: docs locale
	poetry build

check: mypy ruff tests

mypy: $(mypy_targets)

ruff: $(ruff_targets)

ruff-fix: $(ruff_fix_targets)

locale: $(mo_file)

docs:
	. $(venv_bin)/activate && $(MAKE) -C docs html

tests: venv
	. $(venv_bin)/activate && python -m unittest tests/server_test.py

$(pot_file): venv
	pybabel extract --keywords=translate:2 $(tui_module) -o $(pot_file)

$(po_file): $(pot_file)
	mkdir -p $(po_dir)
	touch $(po_file)
	pybabel update -D $(tui_module) -i $(pot_file) -l $(po_locale) -d $(po_base)

$(mo_file): $(po_file)
	pybabel compile -f -D $(tui_module) -l $(po_locale) -d $(po_base)

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
.PHONY: all check build mypy ruff ruff-fix locale docs tests _check_poetry $(mypy_targets) $(ruff_targets) $(ruff_fix_targets) pot_file
