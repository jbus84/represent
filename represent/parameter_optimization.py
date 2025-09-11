"""
Parameter Optimization for Target Generators

This module provides Bayesian optimization for target generator parameters,
optimizing for returns with transaction costs using ReturnsEstimatorWithFees.
"""

import warnings
from typing import Any

import numpy as np
import polars as pl

# Try to import optimization dependencies
try:
    # Apply NumPy 2.x compatibility patch for scikit-optimize
    import numpy as np
    if not hasattr(np, 'int'):
        np.int = int
        np.float = float
    
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Try to import tstrends for returns estimation
try:
    from tstrends.returns_estimation import ReturnsEstimatorWithFees
    from tstrends.returns_estimation.fees_config import FeesConfig
    TSTRENDS_AVAILABLE = True
except ImportError:
    TSTRENDS_AVAILABLE = False

from .target_generators.base import TargetGenerator


class ParameterOptimizer:
    """
    Bayesian parameter optimizer for target generators.

    Uses Gaussian Process optimization to find optimal parameters that maximize
    returns after accounting for transaction costs.
    """

    def __init__(
        self,
        fee_pips: float = 0.7,
        initial_points: int = 10,
        n_calls: int = 50,
        random_state: int | None = None,
        verbose: bool = True
    ):
        """
        Initialize parameter optimizer.

        Args:
            fee_pips: Transaction fee in pips (default: 0.7)
            initial_points: Number of random initial points
            n_calls: Total number of optimization calls
            random_state: Random seed for reproducibility
            verbose: Whether to print optimization progress
        """
        if not OPTIMIZATION_AVAILABLE:
            raise ImportError(
                "Parameter optimization requires scikit-optimize. "
                "Install with: pip install scikit-optimize"
            )

        if not TSTRENDS_AVAILABLE:
            raise ImportError(
                "Returns estimation requires tstrends. "
                "Install with: pip install git+https://github.com/agpenas/tstrends.git"
            )

        self.fee_pips = fee_pips
        self.initial_points = initial_points
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose

    def optimize_ga_labeling(
        self,
        prices: np.ndarray | list[np.ndarray],
        custom_bounds: dict[str, tuple] | None = None
    ) -> dict[str, Any]:
        """
        Optimize GA labeling generator parameters.

        Args:
            prices: Price series or list of price series for optimization
            custom_bounds: Custom parameter bounds (optional)

        Returns:
            Dict with optimal parameters and performance metrics
        """
        from .target_generators.ga_labeling import GALabelingGenerator

        # Default bounds for GA parameters (transaction_cost is fixed at 1 pip)
        default_bounds = {
            'population_size': (20, 80),
            'max_generations': (25, 150),
            'lookforward_window': (1000, 5000),  # Thousands of ticks for meaningful predictions with 1 pip costs
            'min_trades': (10, 100),
            'max_trade_frequency': (0.02, 0.10),  # 2-10% trade frequency (realistic for profitable strategies)
            'min_win_rate': (0.1, 0.4),
            'max_win_rate': (0.6, 0.9),
            'min_profit_factor': (0.5, 2.5),
            'mutation_rate': (0.005, 0.05),
            'crossover_rate': (0.6, 0.95)
        }

        # Update with custom bounds if provided
        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self._optimize_generator(
            GALabelingGenerator,
            prices,
            default_bounds,
            "GA Labeling"
        )

    def optimize_oracle_binary(
        self,
        prices: np.ndarray | list[np.ndarray],
        custom_bounds: dict[str, tuple] | None = None
    ) -> dict[str, Any]:
        """
        Optimize Oracle Binary trend labeling parameters.

        Args:
            prices: Price series or list of price series
            custom_bounds: Custom parameter bounds (optional)

        Returns:
            Dict with optimal parameters and performance metrics
        """
        try:
            from .target_generators.tstrends_labeling import OracleBinaryTrendGenerator
        except ImportError:
            raise ImportError("Oracle labeling requires tstrends integration")

        # Default bounds for Oracle Binary
        default_bounds = {
            'transaction_cost': (0.0, 0.01),
        }

        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self._optimize_generator(
            OracleBinaryTrendGenerator,
            prices,
            default_bounds,
            "Oracle Binary"
        )

    def optimize_oracle_ternary(
        self,
        prices: np.ndarray | list[np.ndarray],
        custom_bounds: dict[str, tuple] | None = None
    ) -> dict[str, Any]:
        """
        Optimize Oracle Ternary trend labeling parameters.

        Args:
            prices: Price series or list of price series
            custom_bounds: Custom parameter bounds (optional)

        Returns:
            Dict with optimal parameters and performance metrics
        """
        try:
            from .target_generators.tstrends_labeling import OracleTernaryTrendGenerator
        except ImportError:
            raise ImportError("Oracle labeling requires tstrends integration")

        # Default bounds for Oracle Ternary
        default_bounds = {
            'transaction_cost': (0.0, 0.01),
            'neutral_reward_factor': (0.0, 1.0),
        }

        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self._optimize_generator(
            OracleTernaryTrendGenerator,
            prices,
            default_bounds,
            "Oracle Ternary"
        )

    def optimize_binary_ctl(
        self,
        prices: np.ndarray | list[np.ndarray],
        custom_bounds: dict[str, tuple] | None = None
    ) -> dict[str, Any]:
        """
        Optimize Binary CTL parameters.

        Args:
            prices: Price series or list of price series
            custom_bounds: Custom parameter bounds (optional)

        Returns:
            Dict with optimal parameters and performance metrics
        """
        try:
            from .target_generators.tstrends_labeling import BinaryCTLGenerator
        except ImportError:
            raise ImportError("CTL labeling requires tstrends integration")

        # Default bounds for Binary CTL (from tstrends documentation)
        default_bounds = {
            'omega': (0.0, 0.01),
        }

        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self._optimize_generator(
            BinaryCTLGenerator,
            prices,
            default_bounds,
            "Binary CTL"
        )

    def optimize_ternary_ctl(
        self,
        prices: np.ndarray | list[np.ndarray],
        custom_bounds: dict[str, tuple] | None = None
    ) -> dict[str, Any]:
        """
        Optimize Ternary CTL parameters.

        Args:
            prices: Price series or list of price series
            custom_bounds: Custom parameter bounds (optional)

        Returns:
            Dict with optimal parameters and performance metrics
        """
        try:
            from .target_generators.tstrends_labeling import TernaryCTLGenerator
        except ImportError:
            raise ImportError("CTL labeling requires tstrends integration")

        # Default bounds for Ternary CTL (optimized for FX-scale data)
        # Testing revealed sweet spot 1e-06 to 1e-05 for balanced ternary classification
        # This range produces 1-30% neutral labels for proper class balance
        default_bounds = {
            'marginal_change_thres': (0.000001, 0.00001),  # 1e-06 to 1e-05: micro-scale balance
            'window_size': (5, 100),  # Small to medium windows for responsive signals
        }

        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self._optimize_generator(
            TernaryCTLGenerator,
            prices,
            default_bounds,
            "Ternary CTL"
        )

    def _optimize_generator(
        self,
        generator_class: type[TargetGenerator],
        prices: np.ndarray | list[np.ndarray],
        bounds: dict[str, tuple],
        method_name: str
    ) -> dict[str, Any]:
        """
        Generic optimization method for any target generator.

        Args:
            generator_class: Target generator class to optimize
            prices: Price data for optimization
            bounds: Parameter bounds dictionary
            method_name: Name for logging

        Returns:
            Optimization results
        """
        # Ensure prices is a list of arrays
        if isinstance(prices, np.ndarray):
            price_series_list = [prices]
        else:
            price_series_list = prices

        # Create optimization space
        param_names = list(bounds.keys())
        dimensions = []

        for param_name in param_names:
            low, high = bounds[param_name]
            if param_name in ['population_size', 'max_generations', 'lookforward_window',
                             'min_trades', 'window_size']:
                # Integer parameters
                dimensions.append(Integer(low, high, name=param_name))
            else:
                # Float parameters
                dimensions.append(Real(low, high, name=param_name))

        if self.verbose:
            print(f"🔍 Optimizing {method_name} parameters...")
            print(f"   Parameter bounds: {bounds}")
            print(f"   Price series: {len(price_series_list)} series")
            print(f"   Optimization calls: {self.n_calls}")

        # Progress tracking
        evaluation_count = [0]  # Use list for mutable reference
        best_return_so_far = [float('-inf')]  # Track best return
        progress_bar = None
        if TQDM_AVAILABLE and self.verbose:
            progress_bar = tqdm(
                total=self.n_calls, 
                desc=f"🔍 Optimizing {method_name}",
                unit="eval",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )

        @use_named_args(dimensions)
        def objective(**params):
            """Objective function to minimize (negative returns)."""
            evaluation_count[0] += 1
            
            # Update main progress bar
            if progress_bar:
                progress_bar.update(1)
            
            try:
                # Create generator with current parameters
                generator_params = params.copy()
                
                # Only add transaction_cost for Oracle generators that need it
                if "Oracle" in generator_class.__name__:
                    generator_params['transaction_cost'] = 0.0001  # Fixed at 1 pip
                
                generator = generator_class(**generator_params)

                total_returns = 0.0
                total_signals = 0  # Track signal count for frequency penalty
                total_samples = 0  # Track total samples for frequency calculation
                valid_series = 0
                series_progress = None

                # Evaluate on all price series with inner progress
                if TQDM_AVAILABLE and len(price_series_list) > 1:
                    series_progress = tqdm(
                        price_series_list, 
                        desc=f"  📊 Series eval {evaluation_count[0]}/{self.n_calls}",
                        leave=False,
                        unit="series"
                    )
                    series_iterator = series_progress
                else:
                    series_iterator = price_series_list

                for price_series in series_iterator:
                    try:
                        # Create DataFrame for generator
                        df = pl.DataFrame({
                            'mid_price': price_series,
                            'timestamp': range(len(price_series))
                        })

                        # Generate targets
                        result = generator.generate_targets(df)
                        
                        # For GA labeling, use long labels (more stable for optimization)
                        # GA generates both long and short labels, but long labels are more straightforward
                        if any('long_labels' in col for col in result.columns):
                            long_col = [col for col in result.columns if 'long_labels' in col][0]
                            labels = result[long_col].to_numpy()
                        else:
                            # Fallback to last column for other generators
                            labels = result[result.columns[-1]].to_numpy()

                        # Calculate returns using ReturnsEstimatorWithFees (1 pip = 0.0001)
                        fee_decimal = 0.0001  # Fixed at 1 pip
                        fees_config = FeesConfig(
                            lp_transaction_fees=fee_decimal,  # Long position transaction fees
                            sp_transaction_fees=fee_decimal,  # Short position transaction fees
                        )
                        returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
                        returns = returns_estimator.estimate_return(
                            price_series.tolist(),
                            labels.astype(int).tolist()
                        )

                        # Track signal frequency for triple methods
                        if "Triple" in generator_class.__name__:
                            # Count non-zero signals (actual trades)
                            signal_count = np.count_nonzero(labels)
                            total_signals += signal_count
                            total_samples += len(labels)

                        total_returns += returns
                        valid_series += 1

                    except Exception as e:
                        if self.verbose:
                            print(f"   Warning: Failed to evaluate series: {e}")
                        continue

                if series_progress:
                    series_progress.close()

                if valid_series == 0:
                    if progress_bar:
                        progress_bar.set_postfix({
                            'status': 'FAILED - no valid series',
                            'params': f"pop={params.get('population_size', '?')}, window={params.get('lookforward_window', '?')}"
                        })
                    return 1000.0  # High penalty for complete failure

                avg_returns = total_returns / valid_series
                
                # Apply signal frequency penalty for triple methods
                final_score = avg_returns
                if "Triple" in generator_class.__name__ and total_samples > 0:
                    signal_frequency = total_signals / total_samples
                    min_frequency = 0.05  # Require at least 5% signal frequency
                    
                    if signal_frequency < min_frequency:
                        # Heavy penalty for sparse signals: penalize by missing frequency
                        frequency_penalty = (min_frequency - signal_frequency) * 10.0  # 10x penalty
                        final_score = avg_returns - frequency_penalty
                        
                        if self.verbose and evaluation_count[0] % 10 == 0:  # Log occasionally
                            print(f"   Signal frequency penalty: {signal_frequency:.3f} < {min_frequency:.3f}, penalty: {frequency_penalty:.4f}")
                
                # Update best return tracking with final score
                if final_score > best_return_so_far[0]:
                    best_return_so_far[0] = final_score
                
                # Update progress bar with current status
                freq_info = ""
                if "Triple" in generator_class.__name__ and total_samples > 0:
                    signal_freq = total_signals / total_samples
                    freq_info = f", freq={signal_freq:.1%}"
                
                if progress_bar:
                    progress_bar.set_postfix({
                        'best_score': f"{best_return_so_far[0]:.4f}",
                        'current': f"{final_score:.4f}",
                        'returns': f"{avg_returns:.4f}{freq_info}",
                        'params': f"window={params.get('lookforward_window', '?')}"
                    })

                # Return negative score (we minimize, but want to maximize score)
                return -final_score

            except Exception as e:
                if series_progress:
                    series_progress.close()
                if progress_bar:
                    progress_bar.set_postfix({
                        'status': f'ERROR: {str(e)[:20]}...',
                        'params': f"pop={params.get('population_size', '?')}"
                    })
                if self.verbose:
                    print(f"   Warning: Parameter evaluation failed: {e}")
                return 1000.0  # High penalty for invalid parameters

        # Run optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=self.n_calls,
                n_initial_points=self.initial_points,
                random_state=self.random_state,
                verbose=False  # Disable skopt verbose to avoid conflicts with tqdm
            )
            
            # Close progress bar
            if progress_bar:
                progress_bar.close()

        # Extract optimal parameters
        optimal_params = {}
        for i, param_name in enumerate(param_names):
            value = result.x[i]
            # Convert numpy types to native Python types for JSON serialization
            if param_name in ['population_size', 'max_generations', 'lookforward_window',
                             'min_trades', 'window_size', 'volatility_window']:
                optimal_params[param_name] = int(value)
            elif param_name in ['normalize_by_volatility', 'adaptive_scaling']:
                # Convert float to boolean
                optimal_params[param_name] = bool(round(value))
            else:
                optimal_params[param_name] = float(value)

        optimal_returns = -result.fun  # Convert back from minimization

        if self.verbose:
            print(f"✅ {method_name} optimization complete!")
            print(f"   Optimal parameters: {optimal_params}")
            print(f"   Maximum returns: {optimal_returns:.4f}")

        return {
            'method': method_name,
            'optimal_params': optimal_params,
            'maximum_returns': optimal_returns,
            'optimization_result': result,
            'fee_pips': 1.0  # Fixed at 1 pip
        }

    def optimize_triple_barrier(
        self,
        prices: np.ndarray | list[np.ndarray],
        custom_bounds: dict[str, tuple] | None = None
    ) -> dict[str, Any]:
        """
        Optimize Triple Barrier parameters.

        Args:
            prices: Price series or list of price series
            custom_bounds: Custom parameter bounds (optional)

        Returns:
            Dict with optimal parameters and performance metrics
        """
        from .target_generators.triple_barrier import TripleBarrierGenerator

        # PROVEN bounds based on successful diagnostic parameters (2K window, 1 pip barriers)
        # Lookforward limited to 5K samples max, sampling windows to 25K max
        # Bounds centered around proven diagnostic values that generated good signals
        default_bounds = {
            'lookforward_window': (1000, 3000),    # 1K-3K ticks: proven range around 2K
            'barrier_width': (0.00005, 0.0002), # 0.5-2 pips: centered around proven 1 pip
            'min_return_threshold': (0.000005, 0.00003),  # Low threshold: don't filter signals
            'volatility_window': (10, 50),       # Faster volatility adaptation
            'normalize_by_volatility': (0, 1),   # Boolean: 0=False, 1=True
        }

        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self._optimize_generator(
            TripleBarrierGenerator,
            prices,
            default_bounds,
            "Triple Barrier"
        )

    def optimize_triple_exceedance(
        self,
        prices: np.ndarray | list[np.ndarray],
        custom_bounds: dict[str, tuple] | None = None
    ) -> dict[str, Any]:
        """
        Optimize Triple Exceedance parameters with multi-objective optimization.
        
        Optimizes for both maximum returns and minimum lookforward window length,
        using transaction cost-scaled barriers.

        Args:
            prices: Price series or list of price series
            custom_bounds: Custom parameter bounds (optional)

        Returns:
            Dict with optimal parameters and performance metrics
        """
        from .target_generators.triple_exceedance import TripleExceedanceGenerator

        # PROVEN bounds based on successful diagnostic parameters (2K window, 3.0 scaling)
        # Lookforward limited to 5K samples max, sampling windows to 25K max
        # Bounds centered around proven diagnostic values that generated good signals
        default_bounds = {
            'lookforward_window': (1000, 3000),  # 1K-3K ticks: proven range around 2K
            'scaling_factor': (2.0, 4.0),     # 2x-4x transaction cost: centered around proven 3x
            'min_exceedance_threshold': (0.1, 0.5),  # Low threshold: allow more signals
            'volatility_window': (10, 50),     # Faster volatility adaptation  
            'window_penalty_weight': (0.1, 0.3),  # Moderate window length penalty
            'balance_weight': (0.2, 0.8),      # Balance constraint for class distribution
            'target_balance_ratio': (0.25, 0.40),  # Target ratio per class (0.33 = perfect balance)
            'adaptive_scaling': (0, 1),        # Boolean: adaptive volatility scaling
        }

        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self._optimize_generator(
            TripleExceedanceGenerator,
            prices,
            default_bounds,
            "Triple Exceedance"
        )


def optimize_all_methods(
    prices: np.ndarray | list[np.ndarray],
    methods: list[str] | None = None,
    **optimizer_kwargs
) -> dict[str, dict[str, Any]]:
    """
    Optimize parameters for multiple labeling methods.

    Args:
        prices: Price series or list of price series
        methods: List of methods to optimize (None for all available)
        **optimizer_kwargs: Arguments for ParameterOptimizer

    Returns:
        Dict mapping method names to optimization results
    """
    if methods is None:
        methods = ['ga_labeling', 'binary_ctl', 'ternary_ctl', 'oracle_binary', 'oracle_ternary', 'triple_barrier', 'triple_exceedance']

    optimizer = ParameterOptimizer(**optimizer_kwargs)
    results = {}

    method_map = {
        'ga_labeling': optimizer.optimize_ga_labeling,
        'binary_ctl': optimizer.optimize_binary_ctl,
        'ternary_ctl': optimizer.optimize_ternary_ctl,
        'oracle_binary': optimizer.optimize_oracle_binary,
        'oracle_ternary': optimizer.optimize_oracle_ternary,
        'triple_barrier': optimizer.optimize_triple_barrier,
        'triple_exceedance': optimizer.optimize_triple_exceedance,
    }

    for method in methods:
        if method in method_map:
            try:
                print(f"\n{'='*60}")
                results[method] = method_map[method](prices)
            except ImportError as e:
                print(f"⚠️  Skipping {method}: {e}")
                continue
            except Exception as e:
                print(f"❌ Failed to optimize {method}: {e}")
                continue
        else:
            print(f"⚠️  Unknown method: {method}")

    return results
