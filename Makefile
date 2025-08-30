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
	@echo "🎯 SYMBOL DATASET BUILDING:"
	@echo "  list-presets           - List all available label set presets"
	@echo "  build-labels           - Interactive label set builder (prompts for configuration)"
	@echo "  build-trading-labels   - Build trading strategy symbol datasets from all DBN files"
	@echo "  build-research-labels  - Build academic research symbol datasets from all DBN files"
	@echo "  build-mfe-labels       - Build MFE analysis symbol datasets from all DBN files"
	@echo "  build-trend-labels     - Build trend analysis symbol datasets from all DBN files" 
	@echo "  build-vol-labels       - Build volatility analysis symbol datasets from all DBN files"
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
	@echo "🎯 Interactive Symbol Dataset Builder"
	@echo "====================================="
	@echo "Choose your approach:"
	@echo "1. Build full symbol datasets from DBN files (recommended)"
	@echo "2. Build simple label sets from parquet files (legacy)"
	@echo ""
	@echo "Available presets:"
	@python scripts/build_label_set.py --list-presets
	@echo ""
	@echo "💡 For full symbol datasets (includes ALL DBN columns + labels):"
	@echo "   make build-mfe-labels     # MFE analysis with all DBN data"
	@echo "   make build-trend-labels   # Trend analysis with all DBN data"  
	@echo "   make build-vol-labels     # Volatility analysis with all DBN data"
	@echo ""
	@echo "📋 For custom symbol datasets:"
	@echo "   python scripts/build_symbol_datasets_from_dbn.py --config your_config.yaml --dbn-dir /path/to/dbn --output-dir /Users/danielfisher/data/databento/symbol_datasets"
	@echo ""
	@echo "📋 For simple label sets (legacy):"
	@echo "   python scripts/build_label_set.py --config your_config.yaml --data your_data.parquet"

build-trading-labels:
	@echo "🎯 Building Trading Strategy Symbol Datasets"
	@echo "============================================="
	@echo "🎯 Optimized for: Systematic trading strategies with risk management"
	@echo "📊 Processing: All DBN files → Symbol datasets with ALL columns + labels"
	python scripts/build_symbol_datasets_from_dbn.py --config configs/label_sets/trading_strategy.yaml --dbn-dir /Users/danielfisher/data/databento/AUDUSD-micro --output-dir /Users/danielfisher/data/databento/symbol_datasets
	@echo "✅ Trading strategy symbol datasets created successfully!"

build-research-labels:
	@echo "🎯 Building Academic Research Symbol Datasets"
	@echo "=============================================="
	@echo "🎯 Optimized for: Comprehensive financial research and academic analysis"
	@echo "📊 Processing: All DBN files → Symbol datasets with ALL columns + labels"
	python scripts/build_symbol_datasets_from_dbn.py --config configs/label_sets/research_academic.yaml --dbn-dir /Users/danielfisher/data/databento/AUDUSD-micro --output-dir /Users/danielfisher/data/databento/symbol_datasets
	@echo "✅ Research symbol datasets created successfully!"

build-mfe-labels:
	@echo "🎯 Building MFE Analysis Symbol Datasets"
	@echo "========================================="
	@echo "🎯 Optimized for: Maximum Favorable Excursion analysis"
	@echo "📊 Processing: All DBN files → Symbol datasets with ALL columns + MFE labels (BPS)"
	python scripts/build_symbol_datasets_from_dbn.py --preset mfe_analysis --dbn-dir /Users/danielfisher/data/databento/AUDUSD-micro --output-dir /Users/danielfisher/data/databento/symbol_datasets
	@echo "✅ MFE analysis symbol datasets created successfully!"

build-trend-labels:
	@echo "🎯 Building Trend Analysis Symbol Datasets" 
	@echo "==========================================="
	@echo "🎯 Optimized for: Trend detection and remaining value analysis"
	@echo "📊 Processing: All DBN files → Symbol datasets with ALL columns + trend labels"
	python scripts/build_symbol_datasets_from_dbn.py --preset trend_analysis --dbn-dir /Users/danielfisher/data/databento/AUDUSD-micro --output-dir /Users/danielfisher/data/databento/symbol_datasets
	@echo "✅ Trend analysis symbol datasets created successfully!"

build-vol-labels:
	@echo "🎯 Building Volatility Analysis Symbol Datasets"
	@echo "==============================================="
	@echo "🎯 Optimized for: Volatility-based risk management and adaptive strategies"
	@echo "📊 Processing: All DBN files → Symbol datasets with ALL columns + volatility labels"
	python scripts/build_symbol_datasets_from_dbn.py --preset volatility_analysis --dbn-dir /Users/danielfisher/data/databento/AUDUSD-micro --output-dir /Users/danielfisher/data/databento/symbol_datasets
	@echo "✅ Volatility analysis symbol datasets created successfully!"