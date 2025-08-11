.PHONY: install test lint typecheck format build clean help test-performance test-fast test-fast-cov test-coverage coverage-report coverage-html test-unit test-e2e test-e2e-fast symbol-split-merge-demo run-fast-demo generate-reports process-dbn process-dbn-fast process-dbn-production process-dbn-demo process-dbn-balanced

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
	@echo "  symbol-split-merge-demo - Run symbol-split-merge architecture demo"
	@echo "  run-fast-demo          - Run fast processing demo"  
	@echo "  generate-reports       - Generate comprehensive HTML reports for all outputs"
	@echo "  process-dbn            - Process DBN files using symbol-split-merge architecture"
	@echo "  process-dbn-fast       - Fast DBN processing with smaller parameters (for testing)"
	@echo "  process-dbn-production - Production DBN processing with optimized parameters"
	@echo "  process-dbn-demo       - Ultra-fast demo with minimal data (fastest, for quick testing)"
	@echo "  process-dbn-balanced   - Balanced processing (good quality, reasonable resources)"

install:
	uv sync --all-extras

check-commit: ## Run pre-commit checks and commit
	.venv/bin/pre-commit run --all-files
	.venv/bin/cz commit --all

symbol-split-merge-demo:
	uv run python examples/symbol_split_merge_demo.py
	@echo "✅ Symbol-split-merge demo complete!"

run-fast-demo:
	uv run python examples/fast_demo.py  
	@echo "✅ Fast demo complete! Check examples/fast_demo_output/"

generate-reports:
	uv run python examples/create_comprehensive_report.py
	@echo "✅ Comprehensive reports generated! Check examples/ subdirectories for HTML reports"

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
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find examples/ -name "outputs" -type d -exec rm -rf {} + 2>/dev/null || true
	find examples/ -name "classified" -type d -exec rm -rf {} + 2>/dev/null || true
	find examples/ -name "*.png" -type f -delete 2>/dev/null || true
	find examples/ -name "*.json" -type f -delete 2>/dev/null || true
	find examples/ -name "*.parquet" -type f -delete 2>/dev/null || true

# DBN Processing Targets
process-dbn: ## Process DBN files with balanced parameters (recommended)
	@echo "🚀 Processing DBN files with symbol-split-merge architecture..."
	@echo "📁 Looking for DBN files in data/ directory..."
	@if [ ! -d "data" ] || [ -z "$$(find data -name '*.dbn*' -type f 2>/dev/null)" ]; then \
		echo "❌ No DBN files found in data/ directory"; \
		echo "   Please place .dbn or .dbn.zst files in the data/ directory"; \
		exit 1; \
	fi
	@echo "⚙️  Configuration: Balanced (samples=50K, lookback=2K, lookforward=2K)"
	uv run python -c "\
import sys; sys.path.insert(0, '.'); \
from pathlib import Path; \
from represent import create_represent_config, DatasetBuildConfig, build_datasets_from_dbn_files; \
config = create_represent_config(currency='AUDUSD', features=['volume'], samples=50000, lookback_rows=2000, lookforward_input=2000, lookforward_offset=200, jump_size=100); \
dataset_config = DatasetBuildConfig(currency='AUDUSD', features=['volume'], min_symbol_samples=10000, force_uniform=True, keep_intermediate=False); \
dbn_files = list(Path('data').glob('*.dbn*')); \
print(f'📄 Found {len(dbn_files)} DBN files: {[f.name for f in dbn_files]}'); \
results = build_datasets_from_dbn_files(config=config, dbn_files=dbn_files, output_dir='examples/balanced_processing_output', dataset_config=dataset_config, verbose=True); \
print(f'\n✅ SUCCESS: Created {results[\"phase_2_stats\"][\"datasets_created\"]} datasets with {results[\"phase_2_stats\"][\"total_samples\"]:,} total samples'); \
print(f'📁 Output: examples/balanced_processing_output/ directory'); \
"
	@echo "✅ DBN processing complete! Check examples/balanced_processing_output/ directory for results"

process-dbn-fast: ## Fast DBN processing with smaller parameters (for testing/development)
	@echo "⚡ Fast DBN processing (optimized for speed and testing)..."
	@if [ ! -d "data" ] || [ -z "$$(find data -name '*.dbn*' -type f 2>/dev/null)" ]; then \
		echo "❌ No DBN files found in data/ directory"; \
		echo "   Please place .dbn or .dbn.zst files in the data/ directory"; \
		exit 1; \
	fi
	@echo "⚙️  Configuration: Fast (samples=25K, lookback=1K, lookforward=1K)"
	uv run python -c "\
import sys; sys.path.insert(0, '.'); \
from pathlib import Path; \
from represent import create_represent_config, DatasetBuildConfig, build_datasets_from_dbn_files; \
config = create_represent_config(currency='AUDUSD', features=['volume'], samples=25000, lookback_rows=1000, lookforward_input=1000, lookforward_offset=100, jump_size=50); \
dataset_config = DatasetBuildConfig(currency='AUDUSD', features=['volume'], min_symbol_samples=5000, force_uniform=True, keep_intermediate=True); \
dbn_files = list(Path('data').glob('*.dbn*')); \
print(f'📄 Found {len(dbn_files)} DBN files: {[f.name for f in dbn_files]}'); \
results = build_datasets_from_dbn_files(config=config, dbn_files=dbn_files, output_dir='examples/fast_output', dataset_config=dataset_config, verbose=True); \
print(f'\n✅ SUCCESS: Created {results[\"phase_2_stats\"][\"datasets_created\"]} datasets with {results[\"phase_2_stats\"][\"total_samples\"]:,} total samples'); \
print(f'📁 Output: examples/fast_output/ directory (includes intermediate files)'); \
"
	@echo "✅ Fast DBN processing complete! Check examples/fast_output/ directory"

process-dbn-production: ## Production DBN processing with optimized parameters (high quality)
	@echo "🏭 Production DBN processing (optimized for quality and comprehensive datasets)..."
	@if [ ! -d "data" ] || [ -z "$$(find data -name '*.dbn*' -type f 2>/dev/null)" ]; then \
		echo "❌ No DBN files found in data/ directory"; \
		echo "   Please place .dbn or .dbn.zst files in the data/ directory"; \
		exit 1; \
	fi
	@echo "⚙️  Configuration: Production (samples=50K, lookback=2K, lookforward=2K, 2 features)"
	uv run python -c "\
import sys; sys.path.insert(0, '.'); \
from pathlib import Path; \
from represent import create_represent_config, DatasetBuildConfig, build_datasets_from_dbn_files; \
config = create_represent_config(currency='AUDUSD', features=['volume', 'variance'], samples=50000, lookback_rows=2000, lookforward_input=2000, lookforward_offset=300, jump_size=100); \
dataset_config = DatasetBuildConfig(currency='AUDUSD', features=['volume', 'variance'], min_symbol_samples=15000, force_uniform=True, keep_intermediate=False); \
dbn_files = list(Path('data').glob('*.dbn*')); \
print(f'📄 Found {len(dbn_files)} DBN files: {[f.name for f in dbn_files]}'); \
results = build_datasets_from_dbn_files(config=config, dbn_files=dbn_files, output_dir='examples/production_output', dataset_config=dataset_config, verbose=True); \
print(f'\n✅ SUCCESS: Created {results[\"phase_2_stats\"][\"datasets_created\"]} datasets with {results[\"phase_2_stats\"][\"total_samples\"]:,} total samples'); \
print(f'📁 Output: examples/production_output/ directory'); \
"
	@echo "✅ Production DBN processing complete! Check examples/production_output/ directory"

process-dbn-demo: ## Ultra-fast DBN processing demo (minimal data for quick demonstration)
	@echo "⚡ Ultra-fast demo processing (minimal data, maximum speed)..."
	@if [ ! -d "data" ] || [ -z "$$(find data -name '*.dbn*' -type f 2>/dev/null)" ]; then \
		echo "❌ No DBN files found in data/ directory"; \
		echo "   Please place .dbn or .dbn.zst files in the data/ directory"; \
		exit 1; \
	fi
	@echo "⚙️  Configuration: Demo (samples=25K, lookback=200, lookforward=200, 1 file only)"
	uv run python -c "\
import sys; sys.path.insert(0, '.'); \
from pathlib import Path; \
from represent import create_represent_config, DatasetBuildConfig, build_datasets_from_dbn_files; \
config = create_represent_config(currency='AUDUSD', features=['volume'], samples=25000, lookback_rows=200, lookforward_input=200, lookforward_offset=50, jump_size=25); \
dataset_config = DatasetBuildConfig(currency='AUDUSD', features=['volume'], min_symbol_samples=1000, force_uniform=True, keep_intermediate=True); \
dbn_files = list(Path('data').glob('*.dbn*'))[:1]; \
print(f'📄 Found {len(dbn_files)} DBN files (using first file for demo): {[f.name for f in dbn_files]}'); \
results = build_datasets_from_dbn_files(config=config, dbn_files=dbn_files, output_dir='examples/demo_output', dataset_config=dataset_config, verbose=True); \
print(f'\n✅ DEMO SUCCESS: Created {results[\"phase_2_stats\"][\"datasets_created\"]} datasets with {results[\"phase_2_stats\"][\"total_samples\"]:,} total samples'); \
print(f'📁 Output: examples/demo_output/ directory (includes intermediate files)'); \
print(f'⏱️  Processing time: {results[\"total_processing_time_seconds\"]:.1f}s'); \
print(f'📈 Processing rate: {results[\"samples_per_second\"]:.0f} samples/sec'); \
"
	@echo "✅ Ultra-fast demo complete! Check examples/demo_output/ directory"

process-dbn-balanced: ## Balanced DBN processing (good quality with reasonable resource usage)
	@echo "⚖️  Balanced DBN processing (optimized for quality/performance balance)..."
	@if [ ! -d "data" ] || [ -z "$$(find data -name '*.dbn*' -type f 2>/dev/null)" ]; then \
		echo "❌ No DBN files found in data/ directory"; \
		echo "   Please place .dbn or .dbn.zst files in the data/ directory"; \
		exit 1; \
	fi
	@echo "⚙️  Configuration: Balanced (samples=50K, lookback=2K, lookforward=2K, 2 features)"
	uv run python -c "\
import sys; sys.path.insert(0, '.'); \
from pathlib import Path; \
from represent import create_represent_config, DatasetBuildConfig, build_datasets_from_dbn_files; \
config = create_represent_config(currency='AUDUSD', features=['volume', 'variance'], samples=50000, lookback_rows=2000, lookforward_input=2000, lookforward_offset=250, jump_size=50); \
dataset_config = DatasetBuildConfig(currency='AUDUSD', features=['volume', 'variance'], min_symbol_samples=25000, force_uniform=True, keep_intermediate=False); \
dbn_files = list(Path('data').glob('*.dbn*')); \
print(f'📄 Found {len(dbn_files)} DBN files: {[f.name for f in dbn_files]}'); \
results = build_datasets_from_dbn_files(config=config, dbn_files=dbn_files, output_dir='examples/balanced_output', dataset_config=dataset_config, verbose=True); \
print(f'\n✅ SUCCESS: Created {results[\"phase_2_stats\"][\"datasets_created\"]} datasets with {results[\"phase_2_stats\"][\"total_samples\"]:,} total samples'); \
print(f'📁 Output: examples/balanced_output/ directory'); \
print(f'⏱️  Processing time: {results[\"total_processing_time_seconds\"]:.1f}s'); \
print(f'📈 Processing rate: {results[\"samples_per_second\"]:.0f} samples/sec'); \
"
	@echo "✅ Balanced DBN processing complete! Check examples/balanced_output/ directory"