PROJECT=$(shell git rev-parse --show-toplevel 2> /dev/null)
DIR=$(shell pwd)
DOCKER_PROTOC=namely/protoc-all:1.25_0
UID=$(shell id -u)
GID=$(shell id -g)

# virtualenv used for pytest
VENV=.venv
$(VENV):
	$(MAKE) test-env-create

.PHONY: clean
clean:
	rm -rf .venv .pytest_cache .coverage

.PHONY: test-env-create
test-env-create: virtualenv-installed
	[ -d $(VENV) ] || virtualenv -p python3.7 $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r ./requirements.test.txt

virtualenv-installed:
	$(PROJECT)/bin/virtualenv_ensure_installed.sh

PHONY: test
test: $(VENV)
	PYTHONPATH=$(shell echo $${PYTHONPATH}):$(DIR) $(VENV)/bin/py.test -vv $(args)


.PHONY: format
format: $(VENV)
	$(VENV)/bin/black posedetect_core

.PHONY: test-all
test-all: test-format test-lint test-types test

.PHONY: test-format
test-format: $(VENV)
	$(VENV)/bin/black --check posedetect_core

.PHONY: test-lint
test-lint: $(VENV)
	$(VENV)/bin/flake8 .

.PHONY: test-types
test-types: $(VENV)
	. $(VENV)/bin/activate && mypy posedetect_core
