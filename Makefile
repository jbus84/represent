.PHONY: install test lint typecheck format build clean help test-performance test-fast test-fast-cov test-coverage coverage-report coverage-html test-unit test-e2e test-e2e-fast generate-visualization performance-analysis

# Default target
help:
	@echo "Available targets:"
	@echo "  install                - Install dependencies and setup environment"
	@echo "  test                   - Run all tests with coverage (requires 80%)"
	@echo "  test-fast              - Run tests excluding performance tests with coverage"
	@echo "  test-fast-cov          - Run fast tests with coverage report (alias for test-fast)"
	@echo "  test-unit              - Run only unit tests"
	@echo "  test-e2e               - Run only end-to-end tests (with real data)"
	@echo "  test-e2e-fast          - Run only fast end-to-end tests"
	@echo "  test-performance       - Run only performance tests"
	@echo "  test-coverage          - Run tests with coverage report"
	@echo "  coverage-report        - Generate coverage report"
	@echo "  coverage-html          - Generate HTML coverage report"
	@echo "  lint                   - Run linting checks"
	@echo "  typecheck              - Run type checking"
	@echo "  format                 - Format code"
	@echo "  build                  - Build package"
	@echo "  clean                  - Clean build artifacts"
	@echo "  check-commit           - Lints and commits"
	@echo "  generate-visualization - Generate the example visualization"
	@echo "  performance-analysis   - Run PyTorch dataloader performance analysis"

install:
	uv sync --all-extras

check-commit: ## Run pre-commit checks and commit
	.venv/bin/pre-commit run --all-files
	.venv/bin/cz commit --all

generate-visualization:
	uv run python examples/generate_visualization.py

performance-analysis:
	uv run python examples/dataloader_performance_demo.py

test:
	uv run pytest --run-performance -v

test-fast:
	uv run pytest -v

test-fast-cov:
	uv run pytest -v

test-unit:
	uv run pytest tests/unit/ -v

test-e2e:
	uv run pytest tests/e2e/ -v

test-e2e-fast:
	uv run pytest tests/e2e/ -v -m "not slow"

test-performance:
	uv run pytest --performance-only --run-performance -v --no-cov

test-coverage:
	uv run pytest -v

coverage-report:
	uv run coverage report

coverage-html:
	uv run coverage html
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	uv run ruff check .

typecheck:
	uv run pyright

format:
	uv run ruff format .

build:
	uv build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete