"""Generate market depth collapse comparison plots with mid-price overlay."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import databento as db
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from represent import MarketDepthProcessor
from represent.configs import MarketDepthProcessorConfig, PriceBinSpec

# Ensure Matplotlib can cache fonts even when HOME is sandboxed
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

DATA_FILE = Path("data/glbx-mdp3-20240405.mbp-10.dbn.zst")
OUTPUT_DIR = Path("outputs/market_depth_collapse")
TIME_BINS = 50
TICKS_PER_BIN = 100
SAMPLE_ROWS = TIME_BINS * TICKS_PER_BIN

FEATURES: dict[str, dict[str, str]] = {
    "volume": {"cmap": "RdBu", "label": "Normalized depth"},
    "trade_counts": {"cmap": "PuOr", "label": "Normalized counts"},
    "variance": {"cmap": "YlGnBu", "label": "Normalized variance"},
}


def load_filtered_ticks(dbn_path: Path, sample_rows: int) -> pl.DataFrame:
    """Load majority symbol ticks and apply basic quality filters."""
    store = db.read_dbn(str(dbn_path))
    df = store.to_df()

    # Focus on the symbol with the largest number of records
    symbol = df["symbol"].value_counts().idxmax()
    df = df[df["symbol"] == symbol]

    # Remove obvious bad ticks (zero or wildly out-of-range prices)
    price_mask = (
        (df["bid_px_00"] > 0)
        & (df["ask_px_00"] > 0)
        & (df["bid_px_00"] > 0.50)
        & (df["bid_px_00"] < 0.80)
        & (df["ask_px_00"] > 0.50)
        & (df["ask_px_00"] < 0.80)
    )
    df = df.loc[price_mask]

    # Preserve chronological order and take the requested sample
    df = df.sort_values("ts_event").iloc[:sample_rows].copy()
    df.reset_index(drop=True, inplace=True)

    return pl.from_pandas(df)


def compute_mid_rows(config: MarketDepthProcessorConfig, time_bins: int) -> np.ndarray:
    """Return the ladder row for the mid price (centered grid)."""

    effective_levels = cast(int, config.effective_price_levels)
    center_row = ((effective_levels - 2) / 2) + 0.5
    return np.full(time_bins, center_row, dtype=float)


def plot_heatmap(
    array: np.ndarray,
    mid_rows: np.ndarray,
    title: str,
    output_path: Path,
    *,
    cmap: str = "RdBu",
    vmin: float = -1.0,
    vmax: float = 1.0,
    colorbar_label: str = "Normalized depth",
) -> None:
    """Render a heatmap and overlay the normalized mid-price trace."""
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        array,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )

    ax.plot(np.arange(len(mid_rows)), mid_rows, color="black", linewidth=1.5)
    ax.set_xlabel("Time bins")
    ax.set_ylabel("Price ladder")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_configs() -> Iterable[tuple[str, MarketDepthProcessorConfig]]:
    """Yield label/config pairs used in the comparison document."""
    yield "baseline_402", MarketDepthProcessorConfig(
        price_range=200,
        samples=SAMPLE_ROWS,
        ticks_per_bin=TICKS_PER_BIN,
    )
    yield "wide_to_100", MarketDepthProcessorConfig(
        price_range=392,
        target_price_levels=100,
        samples=SAMPLE_ROWS,
        ticks_per_bin=TICKS_PER_BIN,
    )
    yield "wide_to_50", MarketDepthProcessorConfig(
        price_range=384,
        target_price_levels=50,
        samples=SAMPLE_ROWS,
        ticks_per_bin=TICKS_PER_BIN,
    )
    yield "pip_bins_variable", MarketDepthProcessorConfig(
        price_range=400,
        samples=SAMPLE_ROWS,
        ticks_per_bin=TICKS_PER_BIN,
        bin_spec=[
            PriceBinSpec(limit_pips=10, bin_size_pips=1),
            PriceBinSpec(limit_pips=20, bin_size_pips=2),
            PriceBinSpec(limit_pips=None, bin_size_pips=3),
        ],
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ticks = load_filtered_ticks(DATA_FILE, SAMPLE_ROWS)

    for label, config in generate_configs():
        for feature, opts in FEATURES.items():
            processor = MarketDepthProcessor(config=config, features=[feature])
            tensor = processor.process(ticks)
            array = tensor if tensor.ndim == 2 else tensor[0]
            mid_rows = compute_mid_rows(config, array.shape[1])
            title = (
                f"{label.replace('_', ' ').title()} – {feature.replace('_', ' ').title()}"
                f" (shape {array.shape[0]}x{array.shape[1]})"
            )
            plot_heatmap(
                array,
                mid_rows,
                title,
                OUTPUT_DIR / f"{label}_{feature}.png",
                cmap=opts["cmap"],
                colorbar_label=opts["label"],
            )


if __name__ == "__main__":
    main()
