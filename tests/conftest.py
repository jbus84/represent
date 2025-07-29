"""
Pytest configuration and shared fixtures.
"""
import pytest


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