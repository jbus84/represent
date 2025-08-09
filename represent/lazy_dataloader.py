"""
Lazy Loading Parquet DataLoader for Market Depth ML

This module provides high-performance lazy loading from parquet datasets
for PyTorch machine learning workflows with market depth data.
"""

from pathlib import Path
from typing import Iterator, Optional, Tuple, Union, List, Dict, Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

from .constants import PRICE_LEVELS
from .config import RepresentConfig


class LazyParquetDataset(Dataset):
    """
    High-performance lazy loading dataset for parquet-stored market depth data.

    Features:
    - Lazy loading with polars for memory efficiency
    - PyTorch-compatible tensor output
    - Automatic feature deserialization
    - Configurable batch sampling strategies
    - Built-in classification label support
    """

    def __init__(
        self,
        config: RepresentConfig,
        parquet_path: Union[str, Path],
        batch_size: int = 32,
        shuffle: bool = True,
        sample_fraction: float = 1.0,
        features_column: str = "market_depth_features",
        labels_column: str = "classification_label",
        shape_column: Optional[str] = "feature_shape",
        cache_size: int = 1000,
        prefetch_batches: int = 5,
    ):
        """
        Initialize lazy parquet dataset.

        Args:
            config: RepresentConfig with currency-specific configuration
            parquet_path: Path to labeled parquet file
            batch_size: Batch size for DataLoader compatibility
            shuffle: Whether to shuffle data indices
            sample_fraction: Fraction of dataset to use (for quick testing)
            features_column: Column containing serialized feature tensors
            labels_column: Column containing classification labels
            shape_column: Column containing tensor shape information
            cache_size: Number of samples to cache in memory
            prefetch_batches: Number of batches to prefetch for performance
        """
        self.config = config
        self.parquet_path = Path(parquet_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_fraction = sample_fraction
        self.features_column = features_column
        self.labels_column = labels_column
        self.shape_column = shape_column
        self.cache_size = cache_size
        self.prefetch_batches = prefetch_batches

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        # Initialize lazy frame for metadata
        self._lazy_frame = pl.scan_parquet(str(self.parquet_path))

        # Get dataset size and setup indices
        self._setup_dataset_indices()

        # Simple LRU cache for loaded samples
        self._sample_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_order: List[int] = []

        print(
            f"📊 Lazy dataset initialized: {len(self.indices):,} samples from {self.parquet_path.name}"
        )

    def _setup_dataset_indices(self):
        """Setup dataset indices with optional sampling and shuffling."""
        # Get total number of samples
        total_samples = self._lazy_frame.select(pl.len()).collect().item()

        # Apply sample fraction
        if self.sample_fraction < 1.0:
            sample_size = int(total_samples * self.sample_fraction)
            if self.shuffle:
                self.indices = np.random.choice(total_samples, sample_size, replace=False)
            else:
                self.indices = np.arange(sample_size)
        else:
            self.indices = np.arange(total_samples)

        # Shuffle indices if requested
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.total_samples = total_samples
        self.active_samples = len(self.indices)

    def __len__(self) -> int:
        """Return number of active samples."""
        return self.active_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with lazy loading.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features_tensor, classification_label)
        """
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.indices)}")

        actual_idx = self.indices[idx]

        # Check cache first
        if actual_idx in self._sample_cache:
            return self._sample_cache[actual_idx]

        # Load sample lazily with optimized column selection
        sample_data = (
            self._lazy_frame.slice(actual_idx, 1)
            .select(
                [self.features_column, self.labels_column, self.shape_column]
                if self.shape_column
                else [self.features_column, self.labels_column]
            )
            .collect()
        )

        if len(sample_data) == 0:
            raise IndexError(f"No data found at index {actual_idx}")

        # Deserialize features tensor
        features_data = sample_data[self.features_column][0]

        # Get tensor shape - optimize shape parsing
        time_bins: int = getattr(self.config, 'time_bins')  # Use getattr to help type checker
        if self.shape_column and self.shape_column in sample_data.columns:
            shape_str = sample_data[self.shape_column][0]
            # Faster shape parsing than eval()
            if shape_str == f"(402, {time_bins})":
                shape = (PRICE_LEVELS, time_bins)
            else:
                shape = eval(shape_str)  # Fallback for other shapes
        else:
            # Default shape for market depth features
            shape = (PRICE_LEVELS, time_bins)

        # Handle different data formats with optimized hex decoding
        if isinstance(features_data, str):
            # Hex-encoded string - convert back to bytes (optimized)
            features_bytes = bytes.fromhex(features_data)
        elif isinstance(features_data, bytes):
            # Already bytes
            features_bytes = features_data
        else:
            raise ValueError(f"Unsupported features data type: {type(features_data)}")

        # Reconstruct tensor from bytes - optimized tensor creation
        features_array = np.frombuffer(features_bytes, dtype=np.float32)
        if features_array.shape[0] != shape[0] * shape[1]:
            features_array = features_array.reshape(shape)
        else:
            features_array = features_array.reshape(shape)

        # Create tensor without copy when possible
        features_tensor = torch.from_numpy(features_array.copy())

        # Get classification label - optimize tensor creation
        label_value = sample_data[self.labels_column][0]
        label_tensor = torch.tensor(label_value, dtype=torch.long)

        # Cache the sample
        self._cache_sample(actual_idx, (features_tensor, label_tensor))

        return features_tensor, label_tensor

    def _cache_sample(self, idx: int, sample: Tuple[torch.Tensor, torch.Tensor]):
        """Cache a sample with LRU eviction."""
        # Remove from cache if it already exists
        if idx in self._sample_cache:
            self._cache_order.remove(idx)

        # Add to cache
        self._sample_cache[idx] = sample
        self._cache_order.append(idx)

        # Evict oldest if cache is full
        while len(self._cache_order) > self.cache_size:
            oldest_idx = self._cache_order.pop(0)
            del self._sample_cache[oldest_idx]

    def get_metadata(self, idx: int) -> dict:
        """
        Get metadata for a specific sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample metadata
        """
        actual_idx = self.indices[idx]

        # Load metadata columns
        metadata_cols = [
            "sample_id",
            "start_timestamp",
            "end_timestamp",
            "target_timestamp",
            "symbol",
            "source_file",
            "date",
            "global_start_idx",
            "global_end_idx",
        ]

        # Filter to only existing columns
        available_cols = self._lazy_frame.columns
        existing_cols = [col for col in metadata_cols if col in available_cols]

        if not existing_cols:
            return {"index": actual_idx}

        sample_metadata = self._lazy_frame.slice(actual_idx, 1).select(existing_cols).collect()

        if len(sample_metadata) == 0:
            return {"index": actual_idx}

        return sample_metadata.to_dicts()[0]

    def get_batch_metadata(self, indices: List[int]) -> pl.DataFrame:
        """
        Get metadata for a batch of samples efficiently.

        Args:
            indices: List of sample indices

        Returns:
            DataFrame with batch metadata
        """
        actual_indices = [self.indices[idx] for idx in indices]

        # Create filter for the specific indices
        filter_expr = pl.col("__index__").is_in(actual_indices)

        # Load metadata for batch
        metadata_cols = [
            "sample_id",
            "start_timestamp",
            "end_timestamp",
            "target_timestamp",
            "symbol",
            "source_file",
            "date",
            "global_start_idx",
            "global_end_idx",
        ]

        available_cols = self._lazy_frame.columns
        existing_cols = [col for col in metadata_cols if col in available_cols]

        if not existing_cols:
            return pl.DataFrame({"index": actual_indices})

        return (
            self._lazy_frame.with_row_count("__index__")
            .filter(filter_expr)
            .select(existing_cols)
            .collect()
        )

    def get_dataset_info(self) -> dict:
        """Get comprehensive dataset information."""
        # Get available columns first (using collect_schema to avoid performance warning)
        try:
            available_columns = self._lazy_frame.collect_schema().names()
        except Exception:
            # Fallback to the old method if collect_schema fails
            available_columns = self._lazy_frame.columns

        # Build dynamic query based on available columns
        query_exprs = [pl.len().alias("total_samples")]

        # Add timestamp info if available
        timestamp_col = None
        for col in ["start_timestamp", "timestamp", "ts_event"]:
            if col in available_columns:
                timestamp_col = col
                break

        if timestamp_col:
            query_exprs.extend(
                [
                    pl.col(timestamp_col).min().alias("min_timestamp"),
                    pl.col(timestamp_col).max().alias("max_timestamp"),
                ]
            )

        # Add symbol info if available
        if "symbol" in available_columns:
            query_exprs.append(pl.col("symbol").n_unique().alias("unique_symbols"))

        # Add label distribution if available
        if self.labels_column in available_columns:
            query_exprs.append(
                pl.col(self.labels_column).value_counts().alias("label_distribution")
            )

        stats = self._lazy_frame.select(query_exprs).collect()

        info = {
            "parquet_file": str(self.parquet_path),
            "file_size_mb": self.parquet_path.stat().st_size / 1024 / 1024,
            "total_samples": stats["total_samples"][0],
            "active_samples": self.active_samples,
            "sample_fraction": self.sample_fraction,
            "cache_size": self.cache_size,
            "cached_samples": len(self._sample_cache),
        }

        # Add optional fields if they exist in stats
        if "min_timestamp" in stats.columns:
            info["min_timestamp"] = stats["min_timestamp"][0]
            info["max_timestamp"] = stats["max_timestamp"][0]

        if "unique_symbols" in stats.columns:
            info["unique_symbols"] = stats["unique_symbols"][0]

        if "label_distribution" in stats.columns:
            info["label_distribution"] = stats["label_distribution"][0]

        return info


class LazyParquetDataLoader:
    """
    High-performance DataLoader for lazy parquet datasets.

    Features:
    - Optimized for PyTorch training workflows
    - Memory-efficient batch loading
    - Built-in performance monitoring
    - Configurable prefetching and caching
    """

    def __init__(
        self,
        config: RepresentConfig,
        parquet_path: Union[str, Path],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        sample_fraction: float = 1.0,
        cache_size: int = 1000,
        pin_memory: bool = False,
    ):
        """
        Initialize lazy parquet dataloader.

        Args:
            config: RepresentConfig with currency-specific configuration
            parquet_path: Path to labeled parquet file
            batch_size: Batch size for training
            shuffle: Whether to shuffle samples
            num_workers: Number of worker processes (0 for main process)
            sample_fraction: Fraction of dataset to use
            cache_size: Sample cache size per worker
            pin_memory: Pin memory for GPU transfer
        """
        self.dataset = LazyParquetDataset(
            config=config,
            parquet_path=parquet_path,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_fraction=sample_fraction,
            cache_size=cache_size,
        )

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,  # Shuffling handled by dataset
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def _collate_fn(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function for market depth tensors."""
        features = []
        labels = []

        for feature_tensor, label_tensor in batch:
            features.append(feature_tensor)
            labels.append(label_tensor)

        # Stack tensors into batches
        features_batch = torch.stack(features)
        labels_batch = torch.stack(labels)

        return features_batch, labels_batch

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over batches."""
        return iter(self.dataloader)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.dataloader)

    def get_dataset_info(self) -> dict:
        """Get dataset information."""
        return self.dataset.get_dataset_info()


def create_parquet_dataloader(
    config: RepresentConfig,
    parquet_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    sample_fraction: float = 1.0,
    **kwargs: Any,
) -> LazyParquetDataLoader:
    """
    Convenience function to create a lazy parquet dataloader.

    Args:
        config: RepresentConfig with currency-specific configuration
        parquet_path: Path to labeled parquet file
        batch_size: Batch size for training
        shuffle: Whether to shuffle samples
        num_workers: Number of worker processes
        sample_fraction: Fraction of dataset to use
        **kwargs: Additional arguments for dataloader

    Returns:
        Configured LazyParquetDataLoader
    """
    return LazyParquetDataLoader(
        config=config,
        parquet_path=parquet_path,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sample_fraction=sample_fraction,
        **kwargs,
    )
