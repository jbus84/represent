"""
Modern PyTorch DataLoader for Market Depth ML

This module provides the new simplified, high-performance dataloader
focused on parquet-based datasets with pre-computed labels.
"""

from pathlib import Path
from typing import Union

# Import the new lazy loading components
from .lazy_dataloader import LazyParquetDataset, LazyParquetDataLoader, create_parquet_dataloader

# Re-export main components for backward compatibility
__all__ = [
    "LazyParquetDataset",
    "LazyParquetDataLoader",
    "create_parquet_dataloader",
    "MarketDepthDataLoader",  # Alias for the main dataloader
]

# Main dataloader alias
MarketDepthDataLoader = LazyParquetDataLoader


def create_market_depth_dataloader(
    parquet_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    sample_fraction: float = 1.0,
    **kwargs,
) -> LazyParquetDataLoader:
    """
    Create a market depth dataloader from a labeled parquet dataset.

    This is the main entry point for creating dataloaders in the new architecture.

    Args:
        parquet_path: Path to labeled parquet file (created by DBN converter)
        batch_size: Batch size for training
        shuffle: Whether to shuffle samples
        num_workers: Number of worker processes
        sample_fraction: Fraction of dataset to use (useful for quick testing)
        **kwargs: Additional arguments

    Returns:
        Configured market depth dataloader

    Example:
        ```python
        from represent import create_market_depth_dataloader

        # Create dataloader from converted parquet file
        dataloader = create_market_depth_dataloader(
            parquet_path="data/audusd_labeled.parquet",
            batch_size=64,
            shuffle=True,
            num_workers=4
        )

        # Use in training loop
        for features, labels in dataloader:
            # features: (batch_size, height, width) market depth tensors
            # labels: (batch_size,) classification labels
            pass
        ```
    """
    return create_parquet_dataloader(
        parquet_path=parquet_path,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sample_fraction=sample_fraction,
        **kwargs,
    )
