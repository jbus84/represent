"""
Pytest configuration and shared fixtures for all tests.
"""
import pytest
import polars as pl
import databento as db
from pathlib import Path
from typing import Optional

# Path to the data directory
DATA_DIR = Path(__file__).parent.parent / "data"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (may be slow)"
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance tests (slow)"
    )
    parser.addoption(
        "--performance-only",
        action="store_true", 
        default=False,
        help="Run only performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and filter items."""
    # Add performance marker to benchmark tests
    for item in items:
        if "benchmark" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
    
    # Filter for performance-only if requested
    if config.getoption("--performance-only"):
        selected = []
        for item in items:
            if "performance" in item.keywords:
                selected.append(item)
        items[:] = selected


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip performance tests unless explicitly requested
    if "performance" in item.keywords:
        if not item.config.getoption("--run-performance"):
            pytest.skip("Performance tests skipped (use --run-performance)")


# E2E Test Fixtures for Real Market Data

@pytest.fixture(scope="session")
def real_market_data() -> Optional[pl.DataFrame]:
    """Load real market data from the databento file."""
    dbn_file = DATA_DIR / "glbx-mdp3-20240403.mbp-10.dbn.zst"
    
    if not dbn_file.exists():
        pytest.skip(f"Real market data file not found: {dbn_file}")
    
    try:
        # Read the databento file directly (no API key needed for local files)
        data = db.read_dbn(dbn_file)
        
        # Convert to pandas DataFrame first
        df_pandas = data.to_df()
        
        # Convert to polars DataFrame
        df = pl.from_pandas(df_pandas)
            
        return df
        
    except Exception as e:
        pytest.skip(f"Could not load real market data: {e}")


@pytest.fixture(scope="session") 
def sample_real_data(real_market_data) -> Optional[pl.DataFrame]:
    """Get a sample of real market data for testing."""
    if real_market_data is None:
        return None
        
    # Take exactly 50,000 records for consistent testing (required by pipeline)
    if len(real_market_data) >= 50000:
        return real_market_data.head(50000)
    else:
        # If we don't have enough data, skip tests that need this fixture
        pytest.skip(f"Real data has only {len(real_market_data)} records, need 50,000")


@pytest.fixture(scope="session")
def small_real_data(real_market_data) -> Optional[pl.DataFrame]:
    """Get a small sample of real market data for quick tests."""
    if real_market_data is None:
        return None
        
    # Take first 5,000 records for quick tests
    return real_market_data.head(5000)