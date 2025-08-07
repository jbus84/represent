.PHONY: install test lint typecheck format build clean help test-performance test-fast test-fast-cov test-coverage coverage-report coverage-html test-unit test-e2e test-e2e-fast comprehensive-demo run-all-examples

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
	@echo "  comprehensive-demo     - Run comprehensive demo with all functionality"
	@echo "  run-all-examples       - Run all examples and generate comprehensive HTML report"

install:
	uv sync --all-extras

check-commit: ## Run pre-commit checks and commit
	.venv/bin/pre-commit run --all-files
	.venv/bin/cz commit --all

comprehensive-demo:
	uv run python examples/comprehensive_demo.py
	@echo "✅ Comprehensive demo complete! View report at comprehensive_demo_output/comprehensive_demo_report.html"

run-all-examples:
	@echo "🚀 Running all examples and generating HTML report..."
	uv run python scripts/run_all_examples.py
	@echo "✅ Report generated! Check examples_report/examples_report.html"

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
	rm -rf examples_report/
	rm -rf comprehensive_demo_output/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find examples/ -name "outputs" -type d -exec rm -rf {} + 2>/dev/null || true
	find examples/ -name "classified" -type d -exec rm -rf {} + 2>/dev/null || true
	find examples/ -name "*.png" -type f -delete 2>/dev/null || true
	find examples/ -name "*.json" -type f -delete 2>/dev/null || true
	find examples/ -name "*.parquet" -type f -delete 2>/dev/null || true