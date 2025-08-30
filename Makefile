.PHONY: install test lint typecheck format build clean help test-performance test-fast test-unit test-e2e check-commit coverage-report coverage-html visualize-approaches build-labels list-presets build-trading-labels build-research-labels build-mfe-labels build-trend-labels build-vol-labels

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
	@echo "🚀 VISUALIZATION:"
	@echo "  visualize-approaches   - Create comprehensive visualization of all labeling approaches"
	@echo ""
	@echo "🎯 LABEL SET BUILDING:"
	@echo "  list-presets           - List all available label set presets"
	@echo "  build-labels           - Interactive label set builder (prompts for configuration)"
	@echo "  build-trading-labels   - Build trading strategy focused label set"
	@echo "  build-research-labels  - Build academic research focused label set"
	@echo "  build-mfe-labels       - Build MFE analysis focused label set"
	@echo "  build-trend-labels     - Build trend analysis focused label set" 
	@echo "  build-vol-labels       - Build volatility analysis focused label set"
	@echo ""

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

# Labeling Approaches Visualization
visualize-approaches:
	@echo "📊 Creating comprehensive labeling approaches visualization..."
	@echo "🎯 Generating all available target types with real market data"
	PYTHONPATH=. python examples/labeling_approaches_visualization.py
	@echo "✅ Visualizations created in examples/:"
	@echo "   • classification_approaches_comparison.png"
	@echo "   • regression_approaches_comparison.png" 
	@echo "   • academic_vs_traditional_comparison.png"
	@echo "   • complete_labeling_overview.png"

# Label Set Building
list-presets:
	@echo "🎯 Available Label Set Presets:"
	@echo "================================"
	python scripts/build_label_set.py --list-presets

build-labels:
	@echo "🎯 Interactive Label Set Builder"
	@echo "================================="
	@echo "Choose your configuration approach:"
	@echo "1. Use a preset configuration (recommended)"
	@echo "2. Create from custom YAML file"
	@echo ""
	@echo "Available presets:"
	@python scripts/build_label_set.py --list-presets
	@echo ""
	@echo "💡 Run one of the specific targets below, or use:"
	@echo "   make build-[preset]-labels"
	@echo ""
	@echo "📋 For custom config:"
	@echo "   python scripts/build_label_set.py --config your_config.yaml"

build-trading-labels:
	@echo "🎯 Building Trading Strategy Label Set"
	@echo "======================================="
	@echo "🎯 Optimized for: Systematic trading strategies with risk management"
	@echo "📊 Includes: Multi-horizon MFE, volatility scaling, trend analysis"
	python scripts/build_label_set.py --config configs/label_sets/trading_strategy.yaml --data /Users/danielfisher/data/databento/AUDUSD-micro/
	@echo "✅ Trading strategy labels built successfully!"

build-research-labels:
	@echo "🎯 Building Academic Research Label Set"
	@echo "========================================"
	@echo "🎯 Optimized for: Comprehensive financial research and academic analysis"
	@echo "📊 Includes: Multi-horizon analysis, trend sensitivities, volatility regimes"
	python scripts/build_label_set.py --config configs/label_sets/research_academic.yaml --data /Users/danielfisher/data/databento/AUDUSD-micro/
	@echo "✅ Research labels built successfully!"

build-mfe-labels:
	@echo "🎯 Building MFE Analysis Label Set"
	@echo "==================================="
	@echo "🎯 Optimized for: Maximum Favorable Excursion analysis"
	@echo "📊 Includes: Buy/sell directional signals with multiple horizons"
	python scripts/build_label_set.py --preset mfe_analysis --data /Users/danielfisher/data/databento/AUDUSD-micro/
	@echo "✅ MFE analysis labels built successfully!"

build-trend-labels:
	@echo "🎯 Building Trend Analysis Label Set" 
	@echo "====================================="
	@echo "🎯 Optimized for: Trend detection and remaining value analysis"
	@echo "📊 Includes: Multi-horizon trend signals and quantile classification"
	python scripts/build_label_set.py --preset trend_analysis --data /Users/danielfisher/data/databento/AUDUSD-micro/
	@echo "✅ Trend analysis labels built successfully!"

build-vol-labels:
	@echo "🎯 Building Volatility Analysis Label Set"
	@echo "=========================================="
	@echo "🎯 Optimized for: Volatility-based risk management and adaptive strategies"
	@echo "📊 Includes: Vol-scaled returns, rolling volatility, adaptive barriers"
	python scripts/build_label_set.py --preset volatility_analysis --data /Users/danielfisher/data/databento/AUDUSD-micro/
	@echo "✅ Volatility analysis labels built successfully!"