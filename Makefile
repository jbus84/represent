.PHONY: install test lint typecheck format build clean help test-performance test-fast test-unit test-e2e demo examples process-production check-commit coverage-report coverage-html

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "📦 SETUP & DEVELOPMENT:"
	@echo "  install                - Install dependencies and setup environment"
	@echo "  clean                  - Clean build artifacts and temporary files"
	@echo "  build                  - Build package for distribution"
	@echo ""
	@echo "🧪 TESTING & QUALITY:"
	@echo "  test                   - Run all tests with coverage (requires 80%)"
	@echo "  test-fast              - Run tests excluding performance tests"
	@echo "  test-unit              - Run only unit tests"
	@echo "  test-e2e               - Run only end-to-end tests"
	@echo "  test-performance       - Run only performance tests"
	@echo "  coverage-report        - Generate coverage report"
	@echo "  coverage-html          - Generate HTML coverage report"
	@echo "  lint                   - Run linting checks"
	@echo "  typecheck              - Run type checking"
	@echo "  format                 - Format code with ruff"
	@echo "  check-commit           - Run pre-commit checks and commit"
	@echo ""
	@echo "🚀 EXAMPLES & DEMOS:"
	@echo "  demo                   - Run complete workflow demo (three core modules)"
	@echo "  examples               - Alias for demo"
	@echo ""
	@echo "🏭 PRODUCTION:"
	@echo "  process-production     - Process AUDUSD-micro data for production ML training"

# Setup & Development
install:
	uv sync --all-extras

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find examples/ -name "*_output" -type d -exec rm -rf {} + 2>/dev/null || true
	find examples/ -name "*.png" -type f -delete 2>/dev/null || true
	find examples/ -name "*.parquet" -type f -delete 2>/dev/null || true
	@echo "🧹 Cleanup complete!"

build:
	uv build

# Testing & Quality
test:
	uv run pytest --cov=represent --cov-report=term-missing --cov-fail-under=80 -v

test-fast:
	uv run pytest -v -m "not performance"

test-unit:
	uv run pytest tests/unit/ -v

test-e2e:
	uv run pytest tests/e2e/ -v

test-performance:
	uv run pytest -v -m "performance" --no-cov

coverage-report:
	uv run coverage report

coverage-html:
	uv run coverage html
	@echo "📊 Coverage report generated in htmlcov/index.html"

lint:
	uv run ruff check .

typecheck:
	uv run pyright

format:
	uv run ruff format .

check-commit:
	.venv/bin/pre-commit run --all-files
	.venv/bin/cz commit --all

# Examples & Demos
demo:
	@echo "🚀 Running Complete Workflow Demo"
	@echo "================================="
	python examples/complete_workflow_demo.py

examples: demo

# Production
process-production:
	@echo "🏭 Processing AUDUSD-micro data for production ML training..."
	@echo "📊 Using first-half training approach to prevent data leakage"
	python scripts/process_production_datasets.py
	@echo "🎉 Production datasets created! Ready for ML training in external repository."