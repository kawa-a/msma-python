# Contributing

## Development contract

1. Preserve the frozen R-compatible S4PCA computation path.
2. Add or update regression tests before changing numerical operations.
3. Do not relax numerical tolerances solely to make tests pass.
4. Keep visualization and model-selection code separate from estimation.
5. Reject unsupported analysis routes explicitly.

## Local checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -v
ruff check src tests
python -m build
python -m twine check dist/*
```
