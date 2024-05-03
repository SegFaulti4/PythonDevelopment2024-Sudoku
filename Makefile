python_bin := python3.10
poetry_version := 1.6.1
venv_bin := venv/bin
modules := clients sudoku

mypy_targets := $(addprefix mypy/,$(modules))
ruff_targets := $(addprefix ruff/,$(modules))

all: project $(mypy_targets) $(ruff_targets)

$(mypy_targets): mypy/%: venv
	$(venv_bin)/mypy $*

$(ruff_targets): ruff/%: venv
	$(venv_bin)/ruff check $*

project: venv pyproject.toml _check_poetry
	. $(venv_bin)/activate && poetry install

venv:
	$(python_bin) -m venv venv

_check_poetry:
	if [ "`poetry --version`" != "Poetry (version ${poetry_version})" ]; then \
  		echo "\033[0;31m\033[1mPoetry version ${poetry_version} is required\033[0m\033[0m" && \
    	echo "\033[0;31m\033[1mYou need to install poetry or update poetry to new version\033[0m\033[0m" && \
		exit 1; \
	fi


.SILENT: _check_poetry
.PHONY: all project _check_poetry $(mypy_targets) $(ruff_targets)
