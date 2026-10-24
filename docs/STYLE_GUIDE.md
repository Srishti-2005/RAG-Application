# FastAPI Code Style & Quality Standards

To maintain a clean and reliable codebase, FastAPI follows strict coding conventions and automated formatting.

## Type Hints & Annotations
- All public functions, methods, and class attributes **must** have explicit Python type annotations.
- Use modern typing constructs compatible with Python 3.8+ (e.g. `typing.Union`, `typing.Optional`, or `Annotated` from `typing_extensions`).
- Run `mypy` to verify static typing:
  ```bash
  mypy fastapi
  ```

## Formatting & Linting
- We use **Ruff** for lightning-fast linting and code formatting (replaces Black, Flake8, and isort).
- Maximum line length is **88 characters**.
- Auto-format your code before committing:
  ```bash
  ruff format .
  ruff check --fix .
  ```
- Alternatively, run the helper scripts:
  ```bash
  bash scripts/format.sh
  bash scripts/lint.sh
  ```

## Pre-commit Hooks
We strongly recommend installing `pre-commit` to catch errors automatically before pushing commits:
```bash
pre-commit install
```
To run pre-commit manually across all files:
```bash
pre-commit run --all-files
```

## Docstrings & Comments
- Write docstrings for all public modules, classes, and helper utilities.
- Use Google docstring style or markdown-compatible docstrings.
- Docstrings are extracted to build the official interactive API reference documentation.

## Git Commit Style
- Write clear, imperative commit messages: `fix: handle None in default query parameter`.
- Allowed prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
