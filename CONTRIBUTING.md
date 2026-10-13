# Contributing to scaLR

## Development setup

```
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
pre-commit install
```

## Tests

pytest, test files live next to the code as `test_*.py`, no separate test directory:

```
pytest                    # run everything
pytest -v -s              # verbose, as CI does
pytest scalr/test_api.py  # a single test file
```

## Formatting / import order

Both enforced in CI (config in `.style.yapf` / `.isort.cfg`), and run automatically by `pre-commit`:

```
yapf -ir --style .style.yapf .
isort . --settings-path .isort.cfg
```

## Docs

Sphinx, published to GitHub Pages on push to `main`:

```
sphinx-build docs _build
```

See [CHANGELOG.md](CHANGELOG.md) for a history of notable changes.
