"""
Represent: High-performance LOB feature extraction package.

This package provides ultra-fast market depth processing capabilities
optimized for real-time trading applications.
"""

__version__: str = "1.0.1"

# Import main API functions
from .pipeline import process_market_data, create_processor, MarketDepthProcessor
from .constants import (
    MICRO_PIP_SIZE, TICKS_PER_BIN, SAMPLES, PRICE_LEVELS, TIME_BINS,
    ASK_PRICE_COLUMNS, BID_PRICE_COLUMNS, ASK_VOL_COLUMNS, BID_VOL_COLUMNS
)
from .dataloader import (
    MarketDepthDataset, HighPerformanceDataLoader, 
    BackgroundBatchProducer, AsyncDataLoader,
    create_streaming_dataloader, create_file_dataloader
)

# Public API
__all__ = [
    "process_market_data",
    "create_processor", 
    "MarketDepthProcessor",
    "MarketDepthDataset",
    "HighPerformanceDataLoader",
    "BackgroundBatchProducer",
    "AsyncDataLoader",
    "create_streaming_dataloader",
    "create_file_dataloader",
    "MICRO_PIP_SIZE",
    "TICKS_PER_BIN", 
    "SAMPLES",
    "PRICE_LEVELS",
    "TIME_BINS",
    "ASK_PRICE_COLUMNS",
    "BID_PRICE_COLUMNS", 
    "ASK_VOL_COLUMNS",
    "BID_VOL_COLUMNS",
]