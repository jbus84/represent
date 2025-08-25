.PHONY: install test lint typecheck format build clean help test-performance test-fast test-unit test-e2e demo examples process-production check-commit coverage-report coverage-html distribution-analysis

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
	@echo "📊 RESEARCH & ANALYSIS:"
	@echo "  distribution-analysis  - Run comprehensive distribution analysis (research only)"
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

# Research & Analysis
distribution-analysis:
	@echo "📊 Running Comprehensive Distribution Analysis"
	@echo "=============================================="
	@echo "⚠️  Research only - Not part of main represent package"
	@echo "🎯 Focus: Tail prediction for financial returns classification"
	cd distributions && python scripts/enhanced_distribution_analyzer.py
	@echo "📁 Reports generated:"
	@echo "   • distributions/html/final_comprehensive_report_2024.html"
	@echo "   • distributions/html/distribution_comparison_report_enhanced.html"
	@echo "   • distributions/html/metrics_explanation_detailed.html"
	@echo "🌟 Recommended approach: Merton Jump Diffusion"

# Merton Jump Diffusion Dataset Creation
create-merton-dataset:
	@echo "🎯 Creating Merton Jump Diffusion Classified Dataset"
	@echo "================================================="
	@echo "🌟 Using optimal distribution approach (68% better tail prediction)"
	@echo "📊 Expected: Tail score ~4.6 vs 14.4 baseline"
	@echo "🎲 Method: Merton Jump Diffusion model for financial returns"
	python create_merton_dataset.py
	@echo "📁 Merton datasets created in: /Users/danielfisher/data/databento/AUDUSD_merton_datasets/"
	@echo "✅ Ready for enhanced ML training with superior tail prediction!"

# Dataset Comparison Assessment  
compare-datasets:
	@echo "🔍 Creating Dataset Comparison Assessment"
	@echo "========================================"
	@echo "📊 Comparing: Quantile (Original) vs Merton Jump Diffusion (New)"
	@echo "🎯 Focus: Classification quality, tail prediction, ML training readiness"
	python create_dataset_comparison.py
	@echo "📋 HTML assessment: distributions/html/dataset_comparison_assessment.html"
	@echo "🌟 View detailed comparison results in your browser!"

# Production
process-production:
	@echo "🏭 Processing AUDUSD-micro data for production ML training..."
	@echo "📊 Using first-half training approach to prevent data leakage"
	python scripts/process_production_datasets.py
	@echo "🎉 Production datasets created! Ready for ML training in external repository."