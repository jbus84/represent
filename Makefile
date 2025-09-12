.PHONY: install test lint typecheck format build clean help test-performance test-fast test-unit test-e2e check-commit coverage-report coverage-html visualize-approaches build-labels list-presets build-trading-labels build-research-labels build-mfe-labels build-trend-labels build-vol-labels build-log-return-labels optimize-parameters optimize-parameters-only create-symbol-inputs run-symbol-optimization generate-optimization-report generate-optimized-classifications generate-ga-classifications generate-ctl-classifications generate-quantile-classifications generate-symbol-classifications complete-optimized-workflow

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
	@echo "  build-log-return-labels - Build log return horizons symbol datasets from all DBN files"
	@echo ""
	@echo "🧬 PARAMETER OPTIMIZATION:"
	@echo "  create-symbol-inputs   - Create clean input datasets (without target columns) for optimization"
	@echo "  optimize-parameters-only - Run ONLY parameter optimization (assumes input datasets exist)"
	@echo "  optimize-parameters    - Run parameter optimization on all symbol datasets"
	@echo "  run-symbol-optimization - Run complete optimization workflow (inputs + optimization)"
	@echo "  generate-optimization-report - Generate optimization report and visualizations"
	@echo ""
	@echo "🎯 OPTIMIZED CLASSIFICATIONS:"
	@echo "  generate-optimized-classifications - Generate all optimized classification datasets"
	@echo "  generate-ga-classifications - Generate GA Labeling classifications for all symbols"
	@echo "  generate-ctl-classifications - Generate Binary/Ternary CTL classifications for all symbols"
	@echo "  generate-quantile-classifications - Generate Quantile classifications for all symbols"
	@echo "  generate-symbol-classifications SYMBOL=<name> - Generate all methods for specific symbol"
	@echo "  complete-optimized-workflow - Run complete workflow: inputs → optimization → classifications → reports"
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
	find scripts/ -name "*_output" -type d -exec rm -rf {} + 2>/dev/null || true
	find scripts/ -name "*.png" -type f -delete 2>/dev/null || true
	find scripts/ -name "*.parquet" -type f -delete 2>/dev/null || true
	@echo "🧹 Cleanup complete!"

build:
	uv build

# Testing & Quality
test:
	uv run pytest --cov=represent --cov-report=term-missing --cov-fail-under=20 -v

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
	PYTHONPATH=. python scripts/labeling_approaches_visualization.py
	@echo "✅ Visualizations created in scripts/:"
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

build-log-return-labels:
	@echo "🎯 Building Log Return Horizons Symbol Datasets"
	@echo "==============================================="
	@echo "🎯 Optimized for: Multi-horizon log return analysis (1k-5k ticks)"
	@echo "📊 Processing: All DBN files → Symbol datasets with ALL columns + log return horizon labels"
	python scripts/build_symbol_datasets_from_dbn.py --preset log_return_horizons --dbn-dir /Users/danielfisher/data/databento/AUDUSD-micro --output-dir /Users/danielfisher/data/databento/symbol_datasets
	@echo "✅ Log return horizons symbol datasets created successfully!"

# Parameter Optimization Workflow
optimize-parameters-only:
	@echo "🧬 Running Parameter Optimization Only"
	@echo "======================================"
	@echo "🎯 Optimizing parameters for existing symbol datasets"
	@echo "⚠️  This assumes input datasets already exist in:"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/inputs/*.parquet"
	@echo ""
	@echo "⏱️  Expected runtime: 2-8 hours depending on dataset sizes"
	@echo "💾 Memory usage: ~2GB peak"
	@echo ""
	@echo "🚀 Starting optimization with adaptive sampling..."
	PYTHONPATH=. python scripts/symbol_optimization_runner.py
	@cp -r outputs/optimization_results /Users/danielfisher/data/databento/symbol_datasets/
	@echo "✅ Parameter optimization complete!"
	@echo ""
	@echo "📊 Results saved to:"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/outputs/optimization_results/OPTIMIZATION_RESULTS.md"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/outputs/optimization_results/parameter_comparison.csv"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/outputs/optimization_results/*.png"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/outputs/optimization_results/optimized_parameters/"

create-symbol-inputs:
	@echo "📂 Creating clean symbol input datasets (without target columns)"
	@echo "================================================================"
	@echo "🔍 Processing raw DBN files to create clean input datasets..."
	@echo "   Using existing DBN processing pipeline with inputs_only preset"
	@echo "   This creates symbol datasets with only market microstructure data"
	@echo "   Perfect for parameter optimization and fresh labeling"
	@echo ""
	python scripts/build_symbol_datasets_from_dbn.py --preset inputs_only --dbn-dir /Users/danielfisher/data/databento/AUDUSD-micro --output-dir /Users/danielfisher/data/databento/symbol_datasets/inputs
	@echo ""
	@echo "📊 Clean symbol input datasets created:"
	@ls -lh /Users/danielfisher/data/databento/symbol_datasets/inputs/*.parquet 2>/dev/null || echo "   (No datasets found)"

optimize-parameters:
	@echo "🧬 Running Parameter Optimization on All Symbol Datasets"
	@echo "========================================================="
	@echo "🎯 This will optimize parameters for each symbol dataset individually"
	@echo "⏱️  Expected runtime: 2-8 hours depending on dataset sizes"
	@echo "💾 Memory usage: ~2GB peak"
	@echo ""
	@echo "🚀 Starting optimization..."
	PYTHONPATH=. python scripts/symbol_optimization_runner.py
	@cp -r outputs/optimization_results /Users/danielfisher/data/databento/symbol_datasets/
	@echo "✅ Parameter optimization complete!"
	@echo ""
	@echo "📊 Results saved to:"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/outputs/optimization_results/OPTIMIZATION_RESULTS.md"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/outputs/optimization_results/parameter_comparison.csv"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/outputs/optimization_results/*.png"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/outputs/optimization_results/optimized_parameters/"

run-symbol-optimization: create-symbol-inputs optimize-parameters
	@echo "🎉 Complete Symbol Optimization Workflow Finished!"
	@echo "=================================================="
	@echo "✅ Clean input datasets created (without target columns)"
	@echo "✅ Parameters optimized for each symbol individually"
	@echo "✅ Comprehensive report generated with visualizations"
	@echo ""
	@echo "📊 Results saved to:"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/inputs/ - Clean input datasets"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/outputs/optimization_results/ - Optimized parameters"
	@echo ""
	@echo "📈 Next steps:"
	@echo "   1. Review outputs/optimization_results/OPTIMIZATION_RESULTS.md"
	@echo "   2. Use optimized parameters for production labeling on clean inputs"
	@echo "   3. Run 'make generate-optimization-report' to update visualizations"

generate-optimization-report:
	@echo "📊 Generating Optimization Report and Visualizations"
	@echo "===================================================="
	@echo "🎯 Creating parameter comparison tables and charts..."
	PYTHONPATH=. python -c "\
from represent.parameter_storage import ParameterStorage; \
storage = ParameterStorage('outputs/optimization_results/optimized_parameters'); \
storage.export_to_markdown('outputs/optimization_results/OPTIMIZATION_RESULTS.md'); \
storage.visualize_parameter_distributions(save_path='outputs/optimization_results/parameter_distributions_all.png'); \
storage.create_returns_comparison(save_path='outputs/optimization_results/returns_comparison.png'); \
print('✅ Report and visualizations updated!')"
	@echo "📄 Updated files:"
	@echo "   • outputs/optimization_results/OPTIMIZATION_RESULTS.md"
	@echo "   • outputs/optimization_results/parameter_distributions_*.png"
	@echo "   • outputs/optimization_results/returns_comparison.png"

# Optimized Classification Generation
generate-optimized-classifications:
	@echo "🎯 Generating Optimized Classification Datasets"
	@echo "=============================================="
	@echo "🔍 Using optimized parameters to create labeled datasets for all classification methods"
	@echo "📊 Methods: GA Labeling, Binary CTL, Ternary CTL, Quantile Classification"
	@echo ""
	PYTHONPATH=. uv run python scripts/generate_optimized_classifications.py
	@echo "✅ Optimized classification datasets generated!"
	@echo ""
	@echo "📊 Generated datasets saved to:"
	@ls -lh /Users/danielfisher/data/databento/symbol_datasets/optimized_classifications/*.parquet 2>/dev/null || echo "   (No datasets found)"
	@echo ""
	@echo "📄 Classification report:"
	@ls -lh /Users/danielfisher/data/databento/symbol_datasets/optimized_classifications/OPTIMIZED_CLASSIFICATIONS_REPORT.md 2>/dev/null || echo "   (Report not found)"

generate-ga-classifications:
	@echo "🧬 Generating GA Labeling Classifications (All Symbols)"
	@echo "====================================================="
	PYTHONPATH=. uv run python scripts/generate_optimized_classifications.py --method ga_labeling
	@echo "✅ GA Labeling classifications generated for all symbols!"

generate-ctl-classifications:
	@echo "📊 Generating CTL Classifications (All Symbols)"
	@echo "=============================================="
	PYTHONPATH=. uv run python scripts/generate_optimized_classifications.py --method binary_ctl
	PYTHONPATH=. uv run python scripts/generate_optimized_classifications.py --method ternary_ctl
	@echo "✅ Binary and Ternary CTL classifications generated for all symbols!"

generate-quantile-classifications:
	@echo "📈 Generating Quantile Classifications (All Symbols)"
	@echo "=================================================="
	PYTHONPATH=. uv run python scripts/generate_optimized_classifications.py --method quantile_classification
	@echo "✅ Quantile classifications generated for all symbols!"

generate-symbol-classifications:
	@echo "🎯 Generate Classifications for Specific Symbol"
	@echo "=============================================="
	@echo "Usage: make generate-symbol-classifications SYMBOL=M6AH5"
	@echo "Available symbols: M6AH5, M6AM4, M6AM5, M6AU4, M6AU5, M6AZ4"
	@echo ""
	$(if $(SYMBOL), \
		PYTHONPATH=. uv run python scripts/generate_optimized_classifications.py --symbol $(SYMBOL), \
		@echo "❌ Please specify SYMBOL=<symbol_name>")

complete-optimized-workflow: create-symbol-inputs optimize-parameters generate-optimized-classifications generate-optimization-report
	@echo "🎉 Complete Optimized Classification Workflow Finished!"
	@echo "======================================================"
	@echo "✅ Clean input datasets created"
	@echo "✅ Parameters optimized for all symbols and methods"
	@echo "✅ Optimized classification datasets generated"
	@echo "✅ Comprehensive reports and visualizations created"
	@echo ""
	@echo "📊 Results available in:"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/inputs/ - Clean inputs"
	@echo "   • /Users/danielfisher/data/databento/symbol_datasets/optimized_classifications/ - Labeled datasets" 
	@echo "   • outputs/optimization_results/ - Parameter analysis and reports"