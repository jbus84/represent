# type: ignore
"""
Parameter Storage and Management for Symbol-Specific Optimization

This module provides utilities to store, load, and visualize optimized parameters
for different symbols and contracts.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and other non-serializable types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            # For complex objects, try to extract simple attributes
            return str(obj)
        return super().default(obj)


class ParameterStorage:
    """Manages storage and retrieval of optimized parameters for symbols."""

    def __init__(self, storage_dir: str | Path = "optimized_parameters"):
        """Initialize parameter storage."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def save_symbol_parameters(
        self,
        symbol: str,
        method: str,
        parameters: dict[str, Any],
        metadata: dict[str, Any] = None
    ) -> Path:
        """
        Save optimized parameters for a specific symbol and method.

        Args:
            symbol: Symbol identifier (e.g., 'M6AH5', 'EURUSD')
            method: Optimization method name
            parameters: Optimized parameters dict
            metadata: Additional metadata (dataset info, optimization stats)

        Returns:
            Path to saved file
        """
        # Create symbol directory
        symbol_dir = self.storage_dir / symbol
        symbol_dir.mkdir(exist_ok=True)

        # Prepare data to save
        data = {
            'symbol': symbol,
            'method': method,
            'optimal_params': parameters,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        # Save to JSON file
        filename = f"{method}_params.json"
        filepath = symbol_dir / filename

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyJSONEncoder)

        return filepath

    def load_symbol_parameters(self, symbol: str, method: str) -> dict[str, Any]:
        """Load parameters for a specific symbol and method."""
        filepath = self.storage_dir / symbol / f"{method}_params.json"

        if not filepath.exists():
            raise FileNotFoundError(f"No parameters found for {symbol}/{method}")

        with open(filepath) as f:
            return json.load(f)

    def get_all_symbols(self) -> list[str]:
        """Get list of all symbols with saved parameters."""
        if not self.storage_dir.exists():
            return []

        return [d.name for d in self.storage_dir.iterdir() if d.is_dir()]

    def get_symbol_methods(self, symbol: str) -> list[str]:
        """Get list of methods for which parameters are saved for a symbol."""
        symbol_dir = self.storage_dir / symbol
        if not symbol_dir.exists():
            return []

        methods = []
        for file in symbol_dir.glob("*_params.json"):
            method = file.stem.replace('_params', '')
            methods.append(method)

        return methods

    def create_parameter_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of parameters across all symbols."""
        all_data = []

        for symbol in self.get_all_symbols():
            for method in self.get_symbol_methods(symbol):
                try:
                    data = self.load_symbol_parameters(symbol, method)

                    row = {
                        'symbol': str(symbol),  # Ensure string type
                        'method': str(method),  # Ensure string type
                        'timestamp': data.get('timestamp', 'N/A')
                    }

                    # Add all parameters as columns
                    params = data.get('optimal_params', {})
                    for param_name, param_value in params.items():
                        row[param_name] = param_value

                    # Add metadata if available
                    metadata = data.get('metadata', {})
                    if 'maximum_returns' in metadata:
                        row['returns'] = metadata['maximum_returns']
                    if 'sampling_stats' in metadata:
                        stats = metadata['sampling_stats']
                        row['dataset_size'] = stats.get('total_dataset_size', 'N/A')
                        row['sample_efficiency'] = stats.get('sample_efficiency_percent', 'N/A')

                    all_data.append(row)

                except Exception as e:
                    print(f"Warning: Failed to load {symbol}/{method}: {e}")
                    continue

        df = pd.DataFrame(all_data)

        # Ensure proper data types
        if not df.empty:
            df['symbol'] = df['symbol'].astype(str)
            df['method'] = df['method'].astype(str)

            # Convert numeric columns
            numeric_cols = ['returns']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def visualize_parameter_distributions(
        self,
        method: str = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Create visualization of parameter distributions across symbols.

        Args:
            method: Specific method to visualize (None for all methods)
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        df = self.create_parameter_comparison_table()

        if df.empty:
            print("No parameter data available for visualization")
            return None

        if method:
            df = df[df['method'] == method]
            if df.empty:
                print(f"No data available for method: {method}")
                return None

        # Get numeric parameter columns
        param_cols = []
        for col in df.columns:
            if col not in ['symbol', 'method', 'timestamp', 'returns', 'dataset_size', 'sample_efficiency']:
                if df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df[col]):
                    param_cols.append(col)

        if not param_cols:
            print("No numeric parameters found for visualization")
            return None

        # Create subplots
        n_params = len(param_cols)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

        # Ensure axes is always a 2D array for consistent indexing
        if n_params == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each parameter
        for i, param in enumerate(param_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            # Create box plot or violin plot depending on data
            if df[param].nunique() > 1:
                if method:
                    # Single method - show distribution across symbols
                    sns.boxplot(data=df, y=param, ax=ax)
                    ax.set_title(f'{param} Distribution\n({method})')
                else:
                    # Multiple methods - group by method
                    sns.boxplot(data=df, x='method', y=param, ax=ax)
                    ax.set_title(f'{param} by Method')
                    ax.tick_params(axis='x', rotation=45)
            else:
                # Single value - show bar chart
                value = df[param].iloc[0]
                ax.bar(['Value'], [value])
                ax.set_title(f'{param} = {value}')

            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        if n_params < n_rows * n_cols:
            for i in range(n_params, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter visualization saved to: {save_path}")

        return fig

    def create_returns_comparison(self, save_path: str = None) -> plt.Figure:
        """Create visualization of returns across symbols and methods."""
        df = self.create_parameter_comparison_table()

        if df.empty or 'returns' not in df.columns:
            print("No returns data available for visualization")
            return None

        # Filter out rows without returns data
        df = df.dropna(subset=['returns'])

        if df.empty:
            print("No valid returns data found")
            return None

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Returns by method
        try:
            sns.boxplot(data=df, x='method', y='returns', ax=ax1)
        except Exception as e:
            print(f"⚠️  Boxplot failed: {e}, using bar plot instead")
            # Fallback to bar plot if boxplot fails
            method_means = df.groupby('method')['returns'].mean()
            ax1.bar(method_means.index, method_means.values)
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Returns')
        ax1.set_title('Returns Distribution by Method')
        ax1.set_ylabel('Returns')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Returns by symbol (if multiple symbols)
        if df['symbol'].nunique() > 1:
            try:
                sns.scatterplot(data=df, x='symbol', y='returns', hue='method', ax=ax2)
            except Exception as e:
                print(f"⚠️  Scatterplot failed: {e}, using simple plot instead")
                # Fallback to simple plot
                for method in df['method'].unique():
                    method_data = df[df['method'] == method]
                    ax2.scatter(method_data['symbol'], method_data['returns'], label=method, s=100)
                ax2.legend()
            ax2.set_title('Returns by Symbol and Method')
            ax2.set_ylabel('Returns')
            ax2.tick_params(axis='x', rotation=45)
        else:
            # Single symbol - show returns by method as bar chart
            method_returns = df.groupby('method')['returns'].mean()
            ax2.bar(method_returns.index, method_returns.values)
            ax2.set_title(f'Returns by Method ({df["symbol"].iloc[0]})')
            ax2.set_ylabel('Returns')
            ax2.tick_params(axis='x', rotation=45)

        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Returns comparison saved to: {save_path}")

        return fig

    def export_to_markdown(self, output_path: str) -> str:
        """
        Export parameter comparison to markdown format.

        Args:
            output_path: Path to save markdown file

        Returns:
            Markdown content as string
        """
        df = self.create_parameter_comparison_table()

        if df.empty:
            content = "# Parameter Optimization Results\n\nNo optimization results available.\n"
            with open(output_path, 'w') as f:
                f.write(content)
            return content

        # Create markdown content
        content = []
        content.append("# Symbol-Specific Parameter Optimization Results")
        content.append("")
        content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")

        # Summary statistics
        content.append("## Summary")
        content.append(f"- **Total symbols optimized**: {df['symbol'].nunique()}")
        content.append(f"- **Methods used**: {', '.join(df['method'].unique())}")
        content.append(f"- **Total optimization runs**: {len(df)}")
        content.append("")

        # Returns summary if available
        if 'returns' in df.columns and not df['returns'].isna().all():
            returns_data = df.dropna(subset=['returns'])
            if not returns_data.empty:
                content.append("## Returns Performance")
                content.append(f"- **Best performing method**: {returns_data.loc[returns_data['returns'].idxmax(), 'method']} ({returns_data['returns'].max():.4f})")
                content.append(f"- **Average returns**: {returns_data['returns'].mean():.4f}")
                content.append(f"- **Returns range**: {returns_data['returns'].min():.4f} to {returns_data['returns'].max():.4f}")
                content.append("")

        # Parameter comparison table
        content.append("## Parameter Comparison Table")
        content.append("")

        # Create a more readable table by rounding numeric values
        display_df = df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == 'float64':
                display_df[col] = display_df[col].round(6)

        # Convert to markdown table
        content.append(display_df.to_markdown(index=False))
        content.append("")

        # Method-specific sections
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            content.append(f"## {method.replace('_', ' ').title()} Results")
            content.append("")
            content.append(f"Optimized for **{len(method_df)}** symbols.")
            content.append("")

            if 'returns' in method_df.columns and not method_df['returns'].isna().all():
                returns_data = method_df.dropna(subset=['returns'])
                if not returns_data.empty:
                    content.append("### Performance Metrics")
                    content.append(f"- **Best returns**: {returns_data['returns'].max():.4f} ({returns_data.loc[returns_data['returns'].idxmax(), 'symbol']})")
                    content.append(f"- **Average returns**: {returns_data['returns'].mean():.4f}")
                    content.append(f"- **Worst returns**: {returns_data['returns'].min():.4f} ({returns_data.loc[returns_data['returns'].idxmin(), 'symbol']})")
                    content.append("")

            # Parameter ranges for this method
            param_cols = [col for col in method_df.columns
                         if col not in ['symbol', 'method', 'timestamp', 'returns', 'dataset_size', 'sample_efficiency']
                         and pd.api.types.is_numeric_dtype(method_df[col])]

            if param_cols:
                content.append("### Parameter Ranges")
                for param in param_cols:
                    param_data = method_df[param].dropna()
                    if len(param_data) > 0:
                        if param_data.nunique() > 1:
                            content.append(f"- **{param}**: {param_data.min():.6g} to {param_data.max():.6g} (avg: {param_data.mean():.6g})")
                        else:
                            content.append(f"- **{param}**: {param_data.iloc[0]:.6g} (constant)")
                content.append("")

        # Footer
        content.append("---")
        content.append("*Generated by Represent Parameter Optimization System*")

        # Join and save
        markdown_content = "\n".join(content)

        with open(output_path, 'w') as f:
            f.write(markdown_content)

        print(f"Parameter optimization results exported to: {output_path}")
        return markdown_content


def save_optimization_results(
    symbol: str,
    results: dict[str, dict[str, Any]],
    storage_dir: str = "optimized_parameters"
) -> list[Path]:
    """
    Save optimization results for a symbol across multiple methods.

    Args:
        symbol: Symbol identifier
        results: Dict mapping method names to optimization results
        storage_dir: Directory to store parameters

    Returns:
        List of saved file paths
    """
    storage = ParameterStorage(storage_dir)
    saved_files = []

    for method, result in results.items():
        if 'optimal_params' in result:
            # Extract parameters and metadata
            parameters = result['optimal_params']
            metadata = {
                'maximum_returns': result.get('maximum_returns'),
                'fee_pips': result.get('fee_pips'),
                'sampling_stats': result.get('sampling_stats'),
                'optimization_calls': result.get('optimization_result', {}).get('func_vals', [])
            }

            # Clean up metadata (remove None values)
            metadata = {k: v for k, v in metadata.items() if v is not None}

            filepath = storage.save_symbol_parameters(symbol, method, parameters, metadata)
            saved_files.append(filepath)
            print(f"✅ Saved {method} parameters for {symbol}: {filepath}")

    return saved_files
