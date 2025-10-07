# Repository Guidelines

## Project Structure & Module Organization
`represent/` holds the core library for target generation, optimization, and shared data structures. Automation scripts sit in `scripts/`, labeling presets in `configs/label_sets/`, and tests in `tests/unit/` with reusable fixtures under `tests/unit/fixtures/`. Keep walkthrough material in `docs/`, `notebooks/`, and `examples/`, and store generated reports or datasets in `outputs/`.

## Build, Test, and Development Commands
Install dependencies with `make install` (wraps `uv sync --all-extras`). Run `make test` for the full pytest suite, `make test-unit` for the focused unit set, and `make test-fast` to skip performance markers. Quality gates include `make lint` (ruff), `make typecheck` (pyright), and `make format`; invoke scripts via `uv run python scripts/<task>.py` to ensure the managed environment is used.

## Coding Style & Naming Conventions
Target Python 3.12 and four-space indentation. Ruff enforces a 100-character line limit alongside pycodestyle, pyflakes, bugbear, comprehension, and pyupgrade lint rules—run `make format` and `make lint` before submitting. Use `snake_case` for modules, files, and functions, reserve `PascalCase` for classes, and mirror the explicit type hints used in `represent/data_structures.py`. YAML configuration files in `configs/label_sets/` should follow existing naming patterns such as `trading_strategy.yaml`.

## Testing Guidelines
Pytest drives testing with an 80% coverage threshold (`pyproject.toml`). Name files `test_<topic>.py` and functions `test_<behavior>` so they auto-discover; apply markers like `@pytest.mark.unit`, `@pytest.mark.performance`, or `@pytest.mark.slow` to align with Make targets. Reuse fixtures from `tests/unit/fixtures/` instead of duplicating setup, and generate HTML coverage via `make coverage-html` when validating new modules.

## Commit & Pull Request Guidelines
Commits follow the Conventional Commits format (`feat:`, `fix:`, `chore:`); `.venv/bin/cz commit` and `make check-commit` both run hooks and prepare messages. Pull requests should summarize the change set, highlight config or data touchpoints (for example updates under `configs/label_sets/`), and link related issues. Record validation steps—tests, coverage deltas, or sample script outputs—and attach screenshots when modifying generated reports or notebooks.

## Configuration & Data Tips
Label presets in `configs/label_sets/` drive commands like `make build-labels`; keep schema and naming consistent so automation picks them up. Heavy datasets remain outside the repo (`/Users/danielfisher/data/...` paths referenced in the Makefile), while lightweight outputs belong in `outputs/`. Persist parameter tuning results to `outputs/optimization_results/` so visualization commands reproduce your findings.
