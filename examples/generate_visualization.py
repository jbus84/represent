"""
Generates a visualization of the market depth representation.

This script loads real market data, processes it using the 'represent' library,
and creates a heatmap of the resulting normalized market depth array.
The output is saved as a PNG image that can be easily viewed.
"""
import databento as db
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from represent.pipeline import process_market_data

def generate_visualization() -> None:
    """
    Loads data, processes it, and generates a heatmap visualization.
    """
    # Load the real market data from the .dbn.zst file
    data: db.DBNStore = db.DBNStore.from_file("data/glbx-mdp3-20240405.mbp-10.dbn.zst")
    df_pandas_raw = data.to_df()
    
    # Ensure we have a DataFrame and filter by symbol using pandas, as in the notebook
    df_pandas_base: pd.DataFrame = df_pandas_raw  # type: ignore[assignment]
    df_pandas: pd.DataFrame = df_pandas_base[df_pandas_base["symbol"] == "M6AM4"]  # type: ignore[assignment]
    
    # Define slicing parameters based on the notebook's logic
    SAMPLES: int = 50000
    OFFSET: int = 120000
    start: int = OFFSET
    stop: int = OFFSET + SAMPLES
    
    if len(df_pandas) < stop:
        raise ValueError(f"Not enough data to generate a visualization. Need {stop} samples, but only have {len(df_pandas)}.")

    # Take the slice with pandas, then convert to polars
    df_slice_pandas: pd.DataFrame = df_pandas.iloc[start:stop]  # type: ignore[assignment]
    df_polars: pl.DataFrame = pl.from_pandas(df_slice_pandas)  # type: ignore[arg-type]

    # Process the data to get the normalized market depth representation
    normed_abs_combined = process_market_data(df_polars)

    # Create a heatmap of the processed data
    plt.figure(figsize=(12, 8)) # type: ignore
    plt.imshow(normed_abs_combined, cmap='RdBu', aspect='auto') # type: ignore
    plt.colorbar(label='Normalized Volume Difference') # type: ignore
    plt.title('Market Depth Representation') # type: ignore
    plt.xlabel('Time Bins') # type: ignore
    plt.ylabel('Price Levels') # type: ignore
    
    # Save the visualization to a file
    plt.savefig('examples/market_depth_visualization.png') # type: ignore
    print("Successfully generated 'examples/market_depth_visualization.png'")

if __name__ == '__main__':
    generate_visualization()
