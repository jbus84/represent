"""
Large-Scale Parameter Optimization for Symbol Datasets

This module provides efficient parameter optimization for datasets with millions of samples
using intelligent sampling strategies and parallel evaluation.
"""

import warnings
from typing import Any, Callable
import numpy as np
import time
import polars as pl
from pathlib import Path

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
    SCIKIT_OPTIMIZE_AVAILABLE = True
except ImportError:
    SCIKIT_OPTIMIZE_AVAILABLE = False

# Try to import Optuna for better optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

OPTIMIZATION_AVAILABLE = SCIKIT_OPTIMIZE_AVAILABLE or OPTUNA_AVAILABLE

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


class EarlyStoppingException(Exception):
    """Custom exception to signal early termination of optimization."""
    pass


class LargeScaleParameterOptimizer:
    """
    Parameter optimizer for large-scale symbol datasets using intelligent sampling.
    
    Key features:
    - Random window sampling from large datasets (24M+ samples)
    - Multiple sampling strategies (uniform, stratified, temporal)
    - Parallel evaluation across multiple windows
    - Memory-efficient processing
    - Separate optimization phase before labeling
    """

    def __init__(
        self,
        window_size: int = 25000,  # USER REQ: Reduced to 25K max sampling window
        n_windows: int = 3,        # STABILITY FIX: Increased from 2 to 3 windows for better averaging
        sampling_strategy: str = "stratified",
        fee_pips: float = 0.7,
        initial_points: int = 10,
        n_calls: int = 50,
        random_state: int | None = None,
        verbose: bool = True,
        # Optimization backend selection
        use_optuna: bool = True,  # Prefer Optuna over scikit-optimize
        # Adaptive sampling parameters
        adaptive_sampling: bool = True,
        min_window_size: int = 15000,  # USER REQ: Reduced to 15K min sampling window  
        max_window_size: int = 25000,  # USER REQ: Reduced to 25K max sampling window
        stabilization_threshold: float = 0.05,
        stabilization_patience: int = 3,
        growth_factor: float = 1.5,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        # Debug logging
        debug_log_path: str | Path | None = None,
    ):
        """
        Initialize large-scale parameter optimizer.

        Args:
            window_size: Initial sampling window size (default: 2k samples)
            n_windows: Initial number of windows per evaluation (default: 2)
            sampling_strategy: "uniform", "stratified", or "temporal" sampling
            fee_pips: Transaction fee in pips (default: 0.7)
            initial_points: Number of random initial points
            n_calls: Total number of optimization calls
            random_state: Random seed for reproducibility
            verbose: Whether to print optimization progress
            adaptive_sampling: Enable adaptive sampling until stabilization (default: True)
            min_window_size: Minimum window size for adaptive sampling (default: 2000)
            max_window_size: Maximum window size for adaptive sampling (default: 20000)
            stabilization_threshold: Parameter change threshold for stability (default: 0.05)
            stabilization_patience: Required stable evaluations before stopping growth (default: 3)
            growth_factor: Window size growth factor when unstable (default: 1.5)
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

        self.window_size = window_size
        self.n_windows = n_windows
        self.sampling_strategy = sampling_strategy
        self.fee_pips = fee_pips
        self.initial_points = initial_points
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        
        # Optimization backend preference
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        if use_optuna and not OPTUNA_AVAILABLE:
            warnings.warn("Optuna not available, falling back to scikit-optimize")
            self.use_optuna = False
        
        # Adaptive sampling parameters
        self.adaptive_sampling = adaptive_sampling
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.stabilization_threshold = stabilization_threshold
        self.stabilization_patience = stabilization_patience
        self.growth_factor = growth_factor
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        # Debug logging path
        self.debug_log_path = Path(debug_log_path) if debug_log_path else None
        
        # Adaptive sampling state
        self.current_window_size = window_size
        self.current_n_windows = n_windows
        self.parameter_history = []
        self.stable_count = 0
        
        # Setup random state
        self.rng = np.random.RandomState(random_state)
        
        # Progress reporting counters
        self._sampling_call_count = 0

    def _run_optuna_optimization(self, original_objective, bounds: dict, method_name: str, 
                                generator_class, prices):
        """Run optimization using Optuna with TPE sampler."""
        if self.verbose:
            print(f"   🎯 Using Optuna TPE optimizer")
            
        # Create study with TPE sampler
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(
                n_startup_trials=self.initial_points,
                n_ei_candidates=24,  # Good default for TPE
                seed=self.random_state
            ),
            study_name=f"{method_name}_optimization"
        )
        
        # Helper: directional PnL for {-1,0,1} label streams (long/short/flat)
        def _estimate_directional_pnl(prices_arr: np.ndarray, labels_arr: np.ndarray, fee: float) -> float:
            pnl = 0.0
            position = 0  # -1 short, 0 flat, 1 long
            for t in range(1, len(prices_arr)):
                ret = (prices_arr[t] - prices_arr[t-1]) / prices_arr[t-1]
                # Change position cost
                if labels_arr[t] != position:
                    if position != 0:  # exiting previous position
                        pnl -= fee
                    if labels_arr[t] != 0:  # entering new position
                        pnl -= fee
                    position = labels_arr[t]
                # Accrue returns
                pnl += ret * position
            return pnl
        
        # Progress tracking with TQDM for inline updates
        evaluation_count = [0]
        best_return = [float('-inf')]
        
        # Create simple progress tracker for inline updates
        progress_bar = None
        start_time = time.time()
        if self.verbose:
            self._tqdm_active = True  # Flag to suppress sampling messages
            print(f"🎯 Starting {method_name} optimization...")
        else:
            self._tqdm_active = False  # Ensure sampling messages work when not verbose
        
        # Create Optuna-compatible objective function (replicating original logic)
        def optuna_objective(trial):
            params = {}
            for param_name, (low, high) in bounds.items():
                if param_name in ['population_size', 'max_generations', 'lookforward_window',
                                 'min_trades', 'window_size']:
                    params[param_name] = trial.suggest_int(param_name, int(low), int(high))
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
            
            evaluation_count[0] += 1
            
            try:
                # Create generator with current parameters + fixed transaction cost (0.7 pips)
                generator_params = params.copy()
                generator_params['transaction_cost'] = 0.00007  # Fixed at 0.7 pips
                
                # Create generator with parameters (filter out incompatible params)
                filtered_params = generator_params.copy()
                
                # Remove transaction_cost for CTL generators that don't use it as a parameter
                # (Transaction costs are still applied during returns evaluation)
                if 'CTL' in generator_class.__name__ and 'transaction_cost' in filtered_params:
                    filtered_params.pop('transaction_cost')
                
                # Generic casting for known integer/boolean parameters
                int_like_params = {
                    'population_size', 'max_generations', 'lookforward_window', 'min_trades',
                    'window_size', 'volatility_window'
                }
                
                casted_params = {}
                for k, v in filtered_params.items():
                    if k in int_like_params and isinstance(v, (int, float)):
                        casted_params[k] = int(v)
                    else:
                        casted_params[k] = v
                
                # Sample multiple windows and evaluate
                current_window_size = self.current_window_size if self.adaptive_sampling else self.window_size
                current_n_windows = self.current_n_windows if self.adaptive_sampling else self.n_windows
                
                windows = self.sample_windows(prices, current_n_windows)
                
                total_pnl = 0.0
                valid_evaluations = 0
                
                for window_prices in windows:
                    try:
                        generator = generator_class(**casted_params)
                        window_df = pl.DataFrame({
                            "ts_event": range(len(window_prices)),
                            "mid_price": window_prices,
                        })
                        
                        targets = generator.generate_targets(window_df)
                        if targets is None or len(targets) == 0:
                            continue

 
                        # Fee is total round-trip cost, so divide by 2 for separate entry/exit charges
                        half_fee = (self.fee_pips * 0.0001) / 2.0

                        if method_name in ['Triple Barrier (Large-Scale)']:

                            labels = targets[f"{generator.target_name}"].to_numpy()

                            window_pnl = ((labels != 0).astype(int) * params["barrier_width"] - half_fee).sum()
                            total_pnl += window_pnl
                            valid_evaluations += 1

                        elif method_name in ['Triple Exceedance (Large-Scale)']:

                            buy_labels = targets[f"{generator.target_name}_long"].to_numpy()
                            sell_labels = targets[f"{generator.target_name}_short"].to_numpy()
                            window_pnl = (buy_labels - half_fee).sum()
                            window_pnl += (sell_labels - half_fee).sum()
                            total_pnl += window_pnl
                            valid_evaluations += 1

                        else: # method_name in ["Oracle Binary (Large-Scale)", "Oracle Ternary (Large-Scale)", "CTL Binary (Large-Scale)", "CTL Ternary (Large-Scale)"]:
                            labels = targets[f"{generator.target_name}"].to_numpy()
                            window_pnl = _estimate_directional_pnl(window_prices, labels, half_fee)
                            total_pnl += window_pnl
                            valid_evaluations += 1
                        
                    except Exception as e:
                        print(f"   ❌ Window evaluation failed: {e}")
                        continue
                
                if valid_evaluations == 0:
                    return 1000.0  # High penalty for failed evaluations
                
                avg_return = total_pnl / valid_evaluations
                
                # Parameter stability check for adaptive sampling
                if self.adaptive_sampling and evaluation_count[0] > 1:
                    self._check_parameter_stability(params)
                
                # Update simple progress display
                current_return = -avg_return  # Minimize negative return
                if -current_return > best_return[0]:
                    best_return[0] = -current_return
                
                # Simple inline progress update
                if self.verbose:
                    elapsed = time.time() - start_time
                    progress_pct = (evaluation_count[0] / self.n_calls) * 100
                    eta = (elapsed / evaluation_count[0]) * (self.n_calls - evaluation_count[0]) if evaluation_count[0] > 0 else 0
                    progress_bar_length = 30
                    filled_length = int(progress_bar_length * evaluation_count[0] // self.n_calls)
                    bar = '█' * filled_length + '░' * (progress_bar_length - filled_length)
                    
                    # Use scientific notation for very small values, regular for larger ones
                    if abs(best_return[0]) < 0.001:
                        best_str = f"{best_return[0]:8.2e}"
                    else:
                        best_str = f"{best_return[0]:7.4f}"
                    
                    print(f"\r🎯 {method_name}: {progress_pct:5.1f}% |{bar}| {evaluation_count[0]:2d}/{self.n_calls} "
                          f"[{elapsed:5.1f}s<{eta:5.1f}s] Best: {best_str}", end='', flush=True)
                
                return current_return
                
            except Exception as e:
                if self.verbose and not getattr(self, '_tqdm_active', False):
                    print(f"   ❌ Parameter evaluation failed: {e}")
                return 1000.0
        
        # Custom callback for early stopping and progress
        early_stop_count = 0
        
        def optuna_callback(study, trial):
            nonlocal early_stop_count
            
            # Check for early stopping based on parameter stability
            if self.early_stopping and self.adaptive_sampling:
                min_trials = max(self.initial_points + 5, 15)
                if len(study.trials) >= min_trials:
                    stop_threshold = self.stabilization_patience + self.early_stopping_patience
                    if self.stable_count >= stop_threshold:
                        early_stop_count += 1
                        if early_stop_count >= 3:  # Stop after 3 consecutive stable checks
                            study.stop()
                            if self.verbose and not getattr(self, '_tqdm_active', False):
                                print(f"   🛑 Early stopping triggered by Optuna after {len(study.trials)} trials")
        
        # Run optimization with quieter output
        try:
            # Suppress Optuna's verbose trial output completely when using manual progress
            if self._tqdm_active:
                optuna.logging.set_verbosity(optuna.logging.ERROR)
            else:
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            study.optimize(
                optuna_objective, 
                n_trials=self.n_calls,
                callbacks=[optuna_callback] if self.early_stopping else None,
                show_progress_bar=False  # Disable Optuna's progress bar to avoid conflicts
            )
        except KeyboardInterrupt:
            if self.verbose and not getattr(self, '_tqdm_active', False):
                print(f"   ⏸️  Optimization interrupted after {len(study.trials)} trials")
        
        # Convert Optuna result to scikit-optimize compatible format
        best_trial = study.best_trial
        result_x = [best_trial.params[name] for name in bounds.keys()]
        result_fun = best_trial.value
        
        # Print clean summary
        if self.verbose:
            print(f"\\n🎯 Optimization Summary:")
            print(f"   Trials completed: {len(study.trials)}")
            # Display return in percentage format - our returns are already properly scaled
            return_pct = -result_fun * 100
            print(f"   Best return: {-result_fun:.4f} ({return_pct:.2f}%)")
            print(f"   Best params: {best_trial.params}")
        
        # Create a simple result object that mimics scikit-optimize result
        class OptunaResult:
            def __init__(self, x, fun, trials):
                self.x = x
                self.fun = fun
                self.func_vals = [t.value for t in trials if t.value is not None]
                self.x_iters = [[t.params[name] for name in bounds.keys()] for t in trials if t.value is not None]
            
            def get(self, key, default=None):
                return getattr(self, key, default)
            
            def __getitem__(self, key):
                return getattr(self, key)
            
            def __contains__(self, key):
                return hasattr(self, key)
        
        return OptunaResult(result_x, result_fun, study.trials)

    def _normalize_labels_for_pnl(self, labels: np.ndarray) -> np.ndarray:
        """
        Normalize labels to {-1, 0, 1} format for PnL calculation.
        
        Different generators use different label encodings:
        - Binary: {0, 1} -> {0, 1} (flat/long)
        - Ternary: {0, 1, 2} -> {-1, 0, 1} (short/flat/long) 
        - Oracle: Various encodings
        
        Args:
            labels: Raw labels from generator
            
        Returns:
            Labels normalized to {-1, 0, 1} format
        """
        unique_labels = np.unique(labels)
        
        if len(unique_labels) == 1:
            # Single label - treat as neutral/flat
            return np.zeros_like(labels, dtype=np.int32)
            
        elif len(unique_labels) == 2:
            # Binary case: assume {0, 1} = {flat, long} or {short, long}
            min_label, max_label = unique_labels
            if min_label == 0 and max_label == 1:
                # Standard binary {0, 1} -> {0, 1} (flat/long)
                return labels.astype(np.int32)
            else:
                # Map to {0, 1} format
                normalized = np.zeros_like(labels, dtype=np.int32)
                normalized[labels == max_label] = 1
                return normalized
                
        elif len(unique_labels) == 3:
            # Ternary case: map to {-1, 0, 1}
            sorted_labels = np.sort(unique_labels)
            low, mid, high = sorted_labels
            
            normalized = np.zeros_like(labels, dtype=np.int32)
            normalized[labels == low] = -1   # Short
            normalized[labels == mid] = 0    # Flat  
            normalized[labels == high] = 1   # Long
            
            return normalized
            
        else:
            # More than 3 labels - use quantile mapping
            normalized = np.zeros_like(labels, dtype=np.int32)
            
            # Map lowest third to -1, middle third to 0, highest third to 1
            p33 = np.percentile(unique_labels, 33.33)
            p66 = np.percentile(unique_labels, 66.67)
            
            normalized[labels <= p33] = -1
            normalized[labels >= p66] = 1
            
            return normalized

    def _check_parameter_stability(self, current_params: dict) -> bool:
        """
        Check if parameter estimates have stabilized.
        
        Args:
            current_params: Current best parameter estimates
            
        Returns:
            True if parameters are stable, False otherwise
        """
        if not self.adaptive_sampling or len(self.parameter_history) < 2:
            return False
            
        # Compare with recent parameter history
        recent_params = self.parameter_history[-1]
        
        # Calculate relative changes for each parameter
        max_change = 0.0
        for param_name in current_params.keys():
            if param_name in recent_params:
                old_val = recent_params[param_name]
                new_val = current_params[param_name]
                
                # Calculate relative change (handle zero values)
                if abs(old_val) > 1e-10:
                    rel_change = abs((new_val - old_val) / old_val)
                else:
                    rel_change = abs(new_val - old_val)
                
                max_change = max(max_change, rel_change)
        
        # Parameters are stable if max change is below threshold
        is_stable = max_change < self.stabilization_threshold
        
        if self.verbose and len(self.parameter_history) > 1:
            window_info = f"window={self.current_window_size}x{self.current_n_windows}"
            if is_stable:
                self.stable_count += 1
                print(f"   📊 Parameters stable ({max_change:.3f} < {self.stabilization_threshold:.3f}) - {window_info} - stable count: {self.stable_count}")
            else:
                self.stable_count = 0  # Reset stable count
                print(f"   📈 Parameters changing ({max_change:.3f} >= {self.stabilization_threshold:.3f}) - {window_info}")
                
            # Show parameter details for first few evaluations
            if len(self.parameter_history) <= 5:
                recent_params = self.parameter_history[-1]
                print(f"   🔍 Current params: pop={recent_params.get('population_size', '?')}, "
                      f"gen={recent_params.get('max_generations', '?')}, "
                      f"window={recent_params.get('lookforward_window', '?')}")
        
        return is_stable

    def _should_increase_sampling(self) -> bool:
        """Check if we should increase sampling size due to parameter instability."""
        if not self.adaptive_sampling:
            return False
            
        # Don't increase if we've hit max window size
        if self.current_window_size >= self.max_window_size:
            if self.verbose and self.stable_count >= self.stabilization_patience:
                print(f"   🎯 Sampling stabilized at maximum size: {self.current_window_size:,} samples")
            return False
            
        # Increase sampling if we haven't been stable for enough evaluations
        should_increase = self.stable_count < self.stabilization_patience
        
        # Log when parameters have stabilized
        if not should_increase and self.verbose and self.stable_count == self.stabilization_patience:
            print(f"   🎯 Parameters stabilized! Stopping adaptive growth at {self.current_window_size:,}x{self.current_n_windows}")
            
            # Show the stabilized parameter values
            if len(self.parameter_history) >= 3:
                recent_params = self.parameter_history[-1]  # Most recent parameters
                print(f"   📊 Stabilized parameters:")
                for param_name, value in recent_params.items():
                    if isinstance(value, float):
                        print(f"      • {param_name}: {value:.4f}")
                    else:
                        print(f"      • {param_name}: {value}")
            else:
                print(f"   📊 Parameter history: {len(self.parameter_history)} evaluations tracked")
            
        return should_increase

    def _increase_sampling(self):
        """Increase window size and/or number of windows for better parameter stability."""
        old_window_size = self.current_window_size
        old_n_windows = self.current_n_windows
        
        # Increase window size first, then number of windows
        if self.current_window_size < self.max_window_size:
            self.current_window_size = min(
                int(self.current_window_size * self.growth_factor),
                self.max_window_size
            )
        elif self.current_n_windows < 8:  # Max reasonable number of windows
            self.current_n_windows = min(self.current_n_windows + 1, 8)
            
        if self.verbose:
            print(f"   🔄 Increasing sampling: {old_window_size}x{old_n_windows} → {self.current_window_size}x{self.current_n_windows}")

    def sample_windows(
        self, 
        prices: np.ndarray,
        n_windows: int | None = None
    ) -> list[np.ndarray]:
        """
        Sample windows from large price series using specified strategy.
        
        Args:
            prices: Full price series (potentially 24M+ samples)
            n_windows: Number of windows to sample (uses self.n_windows if None)
            
        Returns:
            List of sampled price windows
        """
        if n_windows is None:
            n_windows = self.n_windows
            
        total_samples = len(prices)
        
        window_size = self.current_window_size if self.adaptive_sampling else self.window_size
        
        if total_samples <= window_size:
            # If data is smaller than window size, return the full series
            return [prices]
        
        windows = []
        
        if self.sampling_strategy == "uniform":
            # Random uniform sampling across the entire series
            for _ in range(n_windows):
                start_idx = self.rng.randint(0, total_samples - window_size + 1)
                end_idx = start_idx + window_size
                windows.append(prices[start_idx:end_idx])
                
        elif self.sampling_strategy == "stratified":
            # Stratified sampling: divide series into segments and sample from each
            segment_size = total_samples // n_windows
            
            for i in range(n_windows):
                # Sample from each segment
                segment_start = i * segment_size
                segment_end = min((i + 1) * segment_size, total_samples)
                
                # Ensure we can fit a window in this segment
                if segment_end - segment_start >= window_size:
                    max_start = segment_end - window_size
                    start_idx = self.rng.randint(segment_start, max_start + 1)
                    end_idx = start_idx + window_size
                    windows.append(prices[start_idx:end_idx])
                else:
                    # Fallback to taking the available data
                    windows.append(prices[segment_start:segment_end])
                    
        elif self.sampling_strategy == "temporal":
            # Temporal sampling: sample windows with increasing time gaps
            # This captures different market regimes and volatility periods
            step_size = (total_samples - window_size) // (n_windows - 1) if n_windows > 1 else 0
            
            for i in range(n_windows):
                start_idx = min(i * step_size, total_samples - window_size)
                end_idx = start_idx + window_size
                windows.append(prices[start_idx:end_idx])
                
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        if self.verbose:
            self._sampling_call_count += 1
            # Don't print sampling info during optimization when TQDM progress bar is active
            # This prevents interference with inline progress updates
            # Only print sampling info every 10th call when not using TQDM
            if not hasattr(self, '_tqdm_active') or not self._tqdm_active:
                if self._sampling_call_count % 10 == 1:
                    total_samples_used = sum(len(w) for w in windows)
                    coverage = (total_samples_used / total_samples) * 100
                    print(f"   📊 Sampled {len(windows)} windows ({total_samples_used:,} samples, {coverage:.1f}% coverage)")
        else:
            self._sampling_call_count += 1
            # Debug log sampled window indices if logging is enabled
            if self.debug_log_path:
                self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.debug_log_path, 'a') as f:
                    f.write(f"SAMPLED_WINDOWS total={len(windows)} coverage={coverage:.3f}% size={window_size}\n")
            
        return windows

    def optimize_with_sampling(
        self,
        generator_class: type[TargetGenerator],
        prices: np.ndarray | str | Path,
        bounds: dict[str, tuple],
        method_name: str,
        data_loader: Callable | None = None
    ) -> dict[str, Any]:
        """
        Optimize generator parameters using window sampling.
        
        Args:
            generator_class: Target generator class to optimize
            prices: Price array, or path to data file
            bounds: Parameter bounds dictionary  
            method_name: Name for logging
            data_loader: Optional function to load data from file path
            
        Returns:
            Optimization results including sampling statistics
        """
        # Load data if path is provided
        if isinstance(prices, (str, Path)):
            if data_loader is None:
                raise ValueError("data_loader function required when prices is a file path")
            prices = data_loader(prices)
        
        total_samples = len(prices)
        
        if self.verbose:
            print(f"🔍 Large-scale optimization: {method_name}")
            print(f"   📈 Dataset: {total_samples:,} samples ({total_samples/1e6:.1f}M)")
            print(f"   📊 Sampling: {self.current_window_size:,} samples × {self.current_n_windows} windows per trial")
            if bounds:
                print(f"   🔍 Using Optuna TPE optimizer ({self.n_calls} trials)")
                print(f"   🎯 Bounds: {bounds}")
            else:
                print(f"   🎯 Fixed parameters method (single evaluation with 0.7 pip transaction cost)")
        
        # Handle methods with no parameters to optimize (e.g., Oracle with fixed transaction costs)
        if not bounds:
            if self.verbose:
                print(f"🎯 Evaluating {method_name} with fixed parameters...")
            
            # Single evaluation with fixed transaction cost
            try:
                # Create generator with fixed 0.7 pip transaction cost
                generator_params = {'transaction_cost': 0.00007}  # Fixed at 0.7 pips
                
                # Filter out incompatible params for generators that don't use transaction_cost
                filtered_params = generator_params.copy()
                if 'CTL' in generator_class.__name__ and 'transaction_cost' in filtered_params:
                    filtered_params.pop('transaction_cost')
                
                generator = generator_class(**filtered_params)
                
                # Sample windows for evaluation  
                windows = self.sample_windows(prices, self.current_n_windows)
                
                # Evaluate across all windows
                total_pnl = 0.0
                valid_evaluations = 0
                
                for window in windows:
                    window_prices = window
                    if len(window_prices) < 100:  # Skip very small windows
                        continue
                    
                    labels = generator.generate(window_prices)
                    normalized_labels = self._normalize_labels_for_pnl(labels)
                    
                    # Apply transaction costs (already halved for round-trip)
                    half_fee = (self.fee_pips * 0.0001) / 2.0
                    window_pnl = self._estimate_directional_pnl(window_prices, normalized_labels, half_fee)
                    
                    total_pnl += window_pnl
                    valid_evaluations += 1
                
                if valid_evaluations == 0:
                    raise ValueError("No valid windows for evaluation")
                
                avg_return = total_pnl / valid_evaluations
                
                if self.verbose:
                    # Display return in percentage format - our returns are already properly scaled
                    return_pct = avg_return * 100
                    print(f"🎯 Fixed evaluation result: {avg_return:.6f} ({return_pct:.2f}%)")
                
                # Return result in standard format
                return {
                    'optimal_params': generator_params,
                    'maximum_returns': avg_return,
                    'method': method_name.replace(' (Large-Scale)', ''),
                    'sampling_stats': {
                        'total_dataset_size': total_samples,
                        'window_size': self.current_window_size,
                        'windows_per_evaluation': len(windows),
                        'samples_per_evaluation': sum(len(w) for w in windows),
                        'total_samples_evaluated': sum(len(w) for w in windows),
                        'sample_efficiency_percent': (sum(len(w) for w in windows) / total_samples) * 100,
                        'sampling_strategy': self.sampling_strategy,
                        'adaptive_sampling': self.adaptive_sampling,
                        'parameter_evaluations_tracked': 1,  # Single evaluation
                        'stabilization_threshold': self.stabilization_threshold
                    },
                    'optimization_calls': [avg_return]  # Single evaluation
                }
                
            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Fixed parameter evaluation failed: {e}")
                return {
                    'optimal_params': {},
                    'maximum_returns': float('-inf'),
                    'method': method_name.replace(' (Large-Scale)', ''),
                    'error': str(e)
                }

        # Create optimization space
        param_names = list(bounds.keys())
        dimensions = []

        for param_name in param_names:
            low, high = bounds[param_name]
            if param_name in ['population_size', 'max_generations', 'lookforward_window',
                             'min_trades', 'window_size']:
                dimensions.append(Integer(low, high, name=param_name))
            else:
                dimensions.append(Real(low, high, name=param_name))

        # Progress tracking for large-scale optimization
        evaluation_count = [0]
        best_return_so_far = [float('-inf')]
        progress_bar = None

        # Helper: directional PnL for {-1,0,1} label streams (long/short/flat)
        def _estimate_directional_pnl(prices_arr: np.ndarray, labels_arr: np.ndarray, fee: float) -> float:
            pnl = 0.0
            position = 0  # -1 short, 0 flat, 1 long
            for t in range(1, len(prices_arr)):
                ret = (prices_arr[t] - prices_arr[t-1]) / prices_arr[t-1]
                # Change position cost
                if labels_arr[t] != position:
                    if position != 0:  # exiting previous position
                        pnl -= fee
                    if labels_arr[t] != 0:  # entering new position
                        pnl -= fee
                    position = labels_arr[t]
                # Accrue returns
                pnl += ret * position
            return pnl
        # Progress tracking handled by manual progress display in Optuna version

        @use_named_args(dimensions)
        def objective(**params):
            """Objective function using adaptive sampled windows."""
            evaluation_count[0] += 1
            
            # Progress tracking handled by Optuna version with manual display
            
            try:
                # Create generator with current parameters + fixed transaction cost (1 pip)
                generator_params = params.copy()
                generator_params['transaction_cost'] = 0.0001  # Fixed at 1 pip
                
                # Create generator with parameters (filter out incompatible params)
                filtered_params = generator_params.copy()
                
                # Remove transaction_cost for CTL generators that don't use it as a parameter
                # (Transaction costs are still applied during returns evaluation)
                if 'CTL' in generator_class.__name__ and 'transaction_cost' in filtered_params:
                    filtered_params.pop('transaction_cost')
                
                # Generic casting for known integer/boolean parameters
                int_like_params = {
                    'population_size', 'max_generations', 'lookforward_window', 'min_trades',
                    'window_size', 'volatility_window'
                }
                
                casted_params = {}
                for k, v in filtered_params.items():
                    if k in int_like_params or k.endswith('_window'):
                        try:
                            casted_params[k] = int(round(v))
                        except Exception:
                            casted_params[k] = v
                    else:
                        casted_params[k] = v
                
                generator = generator_class(**casted_params)
                
                # Sample windows for this evaluation using adaptive sizes
                windows = self.sample_windows(prices, n_windows=self.current_n_windows)
                
                total_returns = 0.0
                total_signals = 0  # Track signal count for frequency penalty
                total_samples = 0  # Track total samples for frequency calculation
                valid_windows = 0
                
                # Evaluate on all sampled windows
                for window_prices in windows:
                    try:
                        # Create DataFrame for generator
                        df = pl.DataFrame({
                            'mid_price': window_prices,
                            'timestamp': range(len(window_prices))
                        })

                        # Data validation for triple barrier methods
                        if method_name in ['Triple Barrier (Large-Scale)', 'Triple Exceedance (Large-Scale)']:
                            lookforward_window = params.get('lookforward_window', 5000)
                            min_required_samples = lookforward_window + 50  # Buffer for safety
                            
                            if len(window_prices) < min_required_samples:
                                # Skip this window - insufficient data
                                continue

                        # Generate targets
                        result = generator.generate_targets(df)
                        
                        # Debug: log label distribution
                        if self.debug_log_path:
                            unique, counts = np.unique(result[result.columns[-1]].to_numpy(), return_counts=True)
                            with open(self.debug_log_path, 'a') as f:
                                f.write(f"LABELS method={method_name} uniq={list(map(int, unique))} cnt={list(map(int, counts))}\n")
                        
                        # For GA labeling, use long labels (more stable for optimization)
                        # GA generates both long and short labels, but long labels are more straightforward
                        if any('long_labels' in col for col in result.columns):
                            long_col = [col for col in result.columns if 'long_labels' in col][0]
                            labels = result[long_col].to_numpy()
                        else:
                            # Fallback to last column for other generators
                            labels = result[result.columns[-1]].to_numpy()

                        # Calculate returns using ReturnsEstimatorWithFees
                        # Use transaction_cost from params if available, otherwise default to 1 pip
                        fee_decimal = params.get('transaction_cost', 0.0001)
                        fees_config = FeesConfig(
                            lp_transaction_fees=fee_decimal,
                            sp_transaction_fees=fee_decimal,
                        )
                        returns_estimator = ReturnsEstimatorWithFees(fees_config=fees_config)
                        
                        # Convert labels to proper format for returns estimation
                        labels_int = labels.astype(int)
                        
                        # Handle different label formats for ReturnsEstimatorWithFees
                        # ReturnsEstimatorWithFees expects {-1, 0, 1} format:
                        # -1 = short position, 0 = no position, 1 = long position
                        
                        unique_labels = np.unique(labels_int[~np.isnan(labels_int)])
                        
                        if set(unique_labels).issubset({0, 1, 2}):
                            # Ternary labels from CTL generators: {0, 1, 2} → convert to {-1, 0, 1}
                            # 0 (Down/Sell) → -1 (Short), 1 (Neutral) → 0 (Hold), 2 (Up/Buy) → 1 (Long)
                            labels_tstrends = labels_int - 1  # {0,1,2} → {-1,0,1}
                        elif len(unique_labels) == 2 and set(unique_labels).issubset({0, 1}):
                            # Binary labels {0, 1} → convert to {0, 1} (0=no position, 1=long position)  
                            # Note: This assumes binary is long-only strategy
                            labels_tstrends = labels_int
                        else:
                            # Assume already in correct format or handle as-is
                            labels_tstrends = labels_int
                        
                        returns = returns_estimator.estimate_return(
                            window_prices.tolist(),
                            labels_tstrends.tolist()
                        )

                        # Track signal frequency for triple methods
                        if method_name in ['Triple Barrier (Large-Scale)', 'Triple Exceedance (Large-Scale)']:
                            # Count non-zero signals (actual trades)
                            signal_count = np.count_nonzero(labels_int)
                            total_signals += signal_count
                            total_samples += len(labels_int)

                        total_returns += returns
                        valid_windows += 1

                    except Exception as e:
                        if self.verbose:
                            print(f"   ⚠️  Window evaluation failed: {e}")
                        continue

                if valid_windows == 0:
                    return 1000.0  # High penalty for complete failure

                avg_returns = total_returns / valid_windows
                
                # Apply signal frequency penalty for triple methods
                final_score = avg_returns
                if method_name in ['Triple Barrier (Large-Scale)', 'Triple Exceedance (Large-Scale)'] and total_samples > 0:
                    signal_frequency = total_signals / total_samples
                    min_frequency = 0.05  # Require at least 5% signal frequency
                    
                    if signal_frequency < min_frequency:
                        # Heavy penalty for sparse signals: penalize by missing frequency
                        frequency_penalty = (min_frequency - signal_frequency) * 10.0  # 10x penalty
                        final_score = avg_returns - frequency_penalty
                        
                        if self.verbose and evaluation_count[0] % 10 == 0:  # Log occasionally
                            print(f"   Signal frequency penalty: {signal_frequency:.3f} < {min_frequency:.3f}, penalty: {frequency_penalty:.4f}")
                
                # Track parameter history for stability analysis (if adaptive sampling is enabled)
                if self.adaptive_sampling:
                    self.parameter_history.append(params.copy())
                    
                    # Check parameter stability after initial phase
                    if evaluation_count[0] > self.initial_points:
                        self._check_parameter_stability(params)
                        
                        # Now check if we should increase sampling based on updated stability
                        if self._should_increase_sampling():
                            self._increase_sampling()
                        
                        # Check for early stopping
                        if (self.early_stopping and 
                            self.stable_count >= self.stabilization_patience + self.early_stopping_patience):
                            if self.verbose:
                                print(f"   🛑 Early stopping triggered ({self.stable_count} stable evaluations)")
                            # Use a custom exception to signal early termination
                            raise EarlyStoppingException("Parameter optimization converged")
                
                # Update best return tracking with final score
                if final_score > best_return_so_far[0]:
                    best_return_so_far[0] = final_score
                
                # Update progress bar with current status
                freq_info = ""
                if method_name in ['Triple Barrier (Large-Scale)', 'Triple Exceedance (Large-Scale)'] and total_samples > 0:
                    signal_freq = total_signals / total_samples
                    freq_info = f", freq={signal_freq:.1%}"
                
                if progress_bar:
                    progress_bar.set_postfix({
                        'best_score': f"{best_return_so_far[0]:.4f}",
                        'current': f"{final_score:.4f}",
                        'returns': f"{avg_returns:.4f}{freq_info}",
                        'windows': f"{valid_windows}/{len(windows)}",
                        'params': f"window={params.get('lookforward_window', '?')}"
                    })
                
                # Return negative score (minimization problem)
                return -final_score

            except EarlyStoppingException as e:
                # Early stopping triggered - this is expected behavior
                if progress_bar:
                    progress_bar.set_postfix({
                        'status': 'EARLY_STOP',
                        'stable': f"{self.stable_count}"
                    })
                # Return the current best return (we want to stop)
                raise e
                
            except Exception as e:
                if progress_bar:
                    progress_bar.set_postfix({
                        'status': f'ERROR: {str(e)[:20]}...',
                        'eval': f"{evaluation_count[0]}/{self.n_calls}"
                    })
                if self.verbose:
                    print(f"   ❌ Parameter evaluation failed: {e}")
                return 1000.0

        # Early stopping callback  
        def early_stopping_callback(res):
            """Check if we should stop optimization early due to parameter stability."""
            if not self.early_stopping or not self.adaptive_sampling:
                return False
            
            # Require a warmup before considering stability
            min_evals_before_stability = max(self.initial_points + 5, 15)
            total_evals = len(res.x_iters) if hasattr(res, 'x_iters') else evaluation_count[0]
            if total_evals < min_evals_before_stability:
                return False
            
            # Stop if parameters have been stable for patience + extra evaluations
            stop_threshold = self.stabilization_patience + self.early_stopping_patience
            if self.stable_count >= stop_threshold:
                if self.verbose:
                    stable_evals = self.stable_count
                    print(f"   🛑 Early stopping after {total_evals} evaluations ({stable_evals} stable)")
                return True
            return False

        # Run optimization with early stopping handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                if self.use_optuna:
                    # Create Optuna-compatible objective function without the decorator
                    result = self._run_optuna_optimization(
                        objective, bounds, method_name, generator_class, prices
                    )
                else:
                    result = gp_minimize(
                        func=objective,
                        dimensions=dimensions,
                        n_calls=self.n_calls,
                        n_initial_points=self.initial_points,
                        random_state=self.random_state,
                        callback=early_stopping_callback,
                        verbose=False  # Disable skopt verbose to avoid conflicts with tqdm
                    )
            except EarlyStoppingException:
                # Early stopping was triggered - create a partial result
                if self.verbose:
                    print(f"   ✅ Early stopping completed after {evaluation_count[0]} evaluations")
                
                # Create a mock result object with the best parameters found so far
                # We'll use the last parameters from history as the "best" result
                if self.parameter_history:
                    best_params_dict = self.parameter_history[-1]
                    result_x = [best_params_dict[param_name] for param_name in param_names]
                    result_fun = best_return_so_far[0]  # This is the best return found
                    
                    # Create a simple result object that mimics scikit-optimize result
                    class EarlyStopResult:
                        def __init__(self, x, fun):
                            self.x = x
                            self.fun = -fun  # Convert back to minimization format
                            self.func_vals = []  # Empty list for compatibility
                            self.x_iters = []    # Empty list for compatibility
                        
                        def get(self, key, default=None):
                            """Provide dict-like get method for compatibility."""
                            return getattr(self, key, default)
                        
                        def __getitem__(self, key):
                            """Allow dict-like access."""
                            return getattr(self, key)
                        
                        def __contains__(self, key):
                            """Allow 'in' operator."""
                            return hasattr(self, key)
                    
                    result = EarlyStopResult(result_x, result_fun)
                else:
                    # Fallback if no history exists
                    raise RuntimeError("Early stopping triggered but no parameter history available")
            
            # Finish progress display and cleanup flag
            if self.verbose:
                print()  # New line after progress display
                self._tqdm_active = False  # Re-enable sampling messages

        # Extract optimal parameters
        optimal_params = {}
        for i, param_name in enumerate(param_names):
            value = result.x[i]
            if param_name in ['population_size', 'max_generations', 'lookforward_window',
                             'min_trades', 'window_size']:
                optimal_params[param_name] = int(value)
            else:
                optimal_params[param_name] = float(value)

        optimal_returns = -result.fun

        # Calculate effective sample efficiency (use adaptive values if enabled)
        final_window_size = self.current_window_size if self.adaptive_sampling else self.window_size
        final_n_windows = self.current_n_windows if self.adaptive_sampling else self.n_windows
        samples_per_eval = final_window_size * final_n_windows
        total_samples_evaluated = samples_per_eval * self.n_calls
        efficiency = (total_samples_evaluated / total_samples) * 100

        if self.verbose:
            print(f"\n✅ {method_name} large-scale optimization complete!")
            print(f"   🎯 Optimal parameters: {optimal_params}")
            print(f"   📈 Maximum returns: {optimal_returns:.4f}")
            
            if self.adaptive_sampling:
                print(f"   📊 Final sampling: {final_window_size:,} samples × {final_n_windows} windows")
                print(f"   🔄 Parameter stability: {len(self.parameter_history)} evaluations tracked")
                
                # Show when parameters stabilized and if early stopping was used
                if self.stable_count >= self.stabilization_patience:
                    stabilization_point = len(self.parameter_history) - self.stable_count + self.stabilization_patience
                    print(f"   🎯 Parameters stabilized at evaluation {stabilization_point}/{len(self.parameter_history)}")
                    
                    if (self.early_stopping and 
                        self.stable_count >= self.stabilization_patience + self.early_stopping_patience):
                        print(f"   🛑 Early stopping used (saved {self.n_calls - len(self.parameter_history)} evaluations)")
                else:
                    print(f"   ⚠️  Parameters never fully stabilized ({self.stable_count}/{self.stabilization_patience} stable)")
                    
                print(f"   📈 Sample efficiency: {efficiency:.1f}% of dataset used (adaptive)")
            else:
                print(f"   📊 Sample efficiency: {efficiency:.1f}% of dataset used")
            
            print(f"   ⏱️  Effective speedup: ~{total_samples/samples_per_eval:.0f}x faster")

        return {
            'method': method_name,
            'optimal_params': optimal_params,
            'maximum_returns': optimal_returns,
            'optimization_result': result,
            'fee_pips': 1.0,  # Fixed at 1 pip
            'sampling_stats': {
                'total_dataset_size': total_samples,
                'window_size': final_window_size,
                'windows_per_evaluation': final_n_windows,
                'samples_per_evaluation': samples_per_eval,
                'total_samples_evaluated': total_samples_evaluated,
                'sample_efficiency_percent': efficiency,
                'sampling_strategy': self.sampling_strategy,
                'adaptive_sampling': self.adaptive_sampling,
                'parameter_evaluations_tracked': len(self.parameter_history) if self.adaptive_sampling else 0,
                'stabilization_threshold': self.stabilization_threshold if self.adaptive_sampling else None
            }
        }

    def optimize_ga_labeling(
        self,
        prices: np.ndarray | str | Path,
        custom_bounds: dict[str, tuple] | None = None,
        data_loader: Callable | None = None
    ) -> dict[str, Any]:
        """Optimize GA labeling with window sampling."""
        from .target_generators.ga_labeling import GALabelingGenerator

        # OPTIMIZED PARAMETERS: population_size=50, max_generations=75, 
        # lookforward_window=250, transaction_cost=0.00007 (71.34% returns)
        # GA parameters with fixed transaction_cost at 1 pip (0.0001)
        default_bounds = {
            'population_size': (20, 80),
            'max_generations': (25, 150),
            'lookforward_window': (1000, 10000),  # Standardized 1K-10K ticks for fair comparison across methods
            'min_trades': (10, 100),
            'min_win_rate': (0.1, 0.4),
            'max_win_rate': (0.6, 0.9),
            'min_profit_factor': (0.5, 2.5),
            'mutation_rate': (0.005, 0.05),
            'crossover_rate': (0.6, 0.95)
        }

        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self.optimize_with_sampling(
            GALabelingGenerator,
            prices,
            default_bounds,
            "GA Labeling (Large-Scale)",
            data_loader
        )

    def optimize_binary_ctl(
        self,
        prices: np.ndarray | str | Path,
        custom_bounds: dict[str, tuple] | None = None,
        data_loader: Callable | None = None
    ) -> dict[str, Any]:
        """Optimize Binary CTL with window sampling."""
        try:
            from .target_generators.tstrends_labeling import BinaryCTLGenerator
        except ImportError:
            raise ImportError("CTL labeling requires tstrends integration")

        # SENSITIVITY-OPTIMIZED PARAMETERS: omega range validated for 90%+ balance
        # Analysis shows omega 0.0-0.0001 works, fails at >=0.0005 - tightened to sweet spot
        default_bounds = {'omega': (0.0, 0.0001)}  # Validated range for 90.4% balance, avoids 100% imbalanced region
        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self.optimize_with_sampling(
            BinaryCTLGenerator,
            prices,
            default_bounds,
            "Binary CTL (Large-Scale)",
            data_loader
        )

    def optimize_ternary_ctl(
        self,
        prices: np.ndarray | str | Path,
        custom_bounds: dict[str, tuple] | None = None,
        data_loader: Callable | None = None
    ) -> dict[str, Any]:
        """Optimize Ternary CTL with window sampling."""
        try:
            from .target_generators.tstrends_labeling import TernaryCTLGenerator
        except ImportError:
            raise ImportError("CTL labeling requires tstrends integration")

        # MICRO-SCALE OPTIMIZED PARAMETERS: fine-tuned for FX-like data
        # Testing revealed sweet spot between 1e-06 and 5e-06 for balanced labels
        # Range (1e-06, 1e-05) captures 1-30% neutral range for proper ternary classification
        default_bounds = {
            'marginal_change_thres': (0.000001, 0.00001),  # 1e-06 to 1e-05: micro-scale balance
            'window_size': (5, 100),  # Small to medium windows for responsive signals
        }
        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self.optimize_with_sampling(
            TernaryCTLGenerator,
            prices,
            default_bounds,
            "Ternary CTL (Large-Scale)",
            data_loader
        )
    
    def optimize_oracle_binary(
        self,
        prices: np.ndarray | str | Path,
        custom_bounds: dict[str, tuple] | None = None,
        data_loader: Callable | None = None
    ) -> dict[str, Any]:
        """Optimize Oracle Binary with window sampling."""
        try:
            from .target_generators.tstrends_labeling import OracleBinaryTrendGenerator
        except ImportError:
            raise ImportError("Oracle labeling requires tstrends integration")

        # SENSITIVITY-OPTIMIZED Oracle Binary: higher transaction costs improve balance
        # Analysis shows TC 0.0001 gives 60% balance vs 27% at lower costs
        default_bounds = {
            'transaction_cost': (0.00001, 0.0001),  # Sweet spot range for best balance (27-60%)
        }
        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self.optimize_with_sampling(
            OracleBinaryTrendGenerator,
            prices,
            default_bounds,
            "Oracle Binary (Large-Scale)",
            data_loader
        )
    
    def optimize_oracle_ternary(
        self,
        prices: np.ndarray | str | Path,
        custom_bounds: dict[str, tuple] | None = None,
        data_loader: Callable | None = None
    ) -> dict[str, Any]:
        """Optimize Oracle Ternary with window sampling."""
        try:
            from .target_generators.tstrends_labeling import OracleTernaryTrendGenerator
        except ImportError:
            raise ImportError("Oracle labeling requires tstrends integration")

        # SENSITIVITY-OPTIMIZED Oracle Ternary: focused on optimal neutral reward factor
        # Analysis shows NRF 0.5 gives best balance (21.5%), transaction cost less sensitive
        default_bounds = {
            'transaction_cost': (0.00001, 0.0001),  # Align with Binary Oracle range
            'neutral_reward_factor': (0.3, 0.7),  # Focused around optimal 0.5 value
        }
        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self.optimize_with_sampling(
            OracleTernaryTrendGenerator,
            prices,
            default_bounds,
            "Oracle Ternary (Large-Scale)",
            data_loader
        )

    def optimize_triple_barrier(
        self,
        prices: np.ndarray | str | Path,
        custom_bounds: dict[str, tuple] | None = None,
        data_loader: Callable | None = None
    ) -> dict[str, Any]:
        """Optimize Triple Barrier with window sampling."""
        from .target_generators.triple_barrier import TripleBarrierGenerator

        # PROVEN bounds based on successful diagnostic parameters (2K window, 1 pip barriers)
        # Lookforward limited to 5K samples max, sampling windows to 25K max
        # Bounds centered around proven diagnostic values that generated good signals
        default_bounds = {
            'lookforward_window': (1000, 3000),    # 1K-3K ticks: proven range around 2K
            'barrier_width': (0.0003, 0.001), # 3-10 pips
        }

        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self.optimize_with_sampling(
            TripleBarrierGenerator,
            prices,
            default_bounds,
            "Triple Barrier (Large-Scale)",
            data_loader
        )

    def optimize_triple_exceedance(
        self,
        prices: np.ndarray | str | Path,
        custom_bounds: dict[str, tuple] | None = None,
        data_loader: Callable | None = None
    ) -> dict[str, Any]:
        """Optimize Triple Exceedance with window sampling and multi-objective optimization."""
        from .target_generators.triple_exceedance import TripleExceedanceGenerator

        # PROVEN bounds based on successful diagnostic parameters (2K window, 3.0 scaling)
        # Lookforward limited to 5K samples max, sampling windows to 25K max
        # Bounds centered around proven diagnostic values that generated good signals
        default_bounds = {
            'lookforward_window': (1000, 5000),    # 1K-3K ticks: proven range around 2K
            'scaling_factor': (2.0, 7.0),       # 2x-4x transaction cost: centered around proven 3x
        }

        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self.optimize_with_sampling(
            TripleExceedanceGenerator,
            prices,
            default_bounds,
            "Triple Exceedance (Large-Scale)",
            data_loader
        )


def optimize_on_symbol_dataset(
    dataset_path: str | Path,
    methods: list[str] | None = None,
    window_size: int = 25000,   # USER REQ: Reduced to 25K max sampling window
    n_windows: int = 3,         # Good: 3 windows for averaging
    sampling_strategy: str = "stratified",
    data_loader: Callable | None = None,
    **optimizer_kwargs
) -> dict[str, dict[str, Any]]:
    """
    Optimize parameters on large symbol datasets using intelligent sampling.
    
    Args:
        dataset_path: Path to symbol dataset file
        methods: Methods to optimize (None for all available)
        window_size: Size of sampling windows
        n_windows: Number of windows per evaluation
        sampling_strategy: Sampling strategy ("uniform", "stratified", "temporal")
        data_loader: Function to load data from file path
        **optimizer_kwargs: Additional optimizer arguments
        
    Returns:
        Dict mapping method names to optimization results
    """
    if methods is None:
        methods = ['ga_labeling', 'binary_ctl', 'ternary_ctl', 'oracle_binary', 'oracle_ternary', 'triple_barrier', 'triple_exceedance']

    optimizer = LargeScaleParameterOptimizer(
        window_size=window_size,
        n_windows=n_windows,
        sampling_strategy=sampling_strategy,
        **optimizer_kwargs
    )
    
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
                print(f"\n{'='*80}")
                results[method] = method_map[method](
                    dataset_path, 
                    data_loader=data_loader
                )
            except ImportError as e:
                print(f"⚠️  Skipping {method}: {e}")
                continue
            except Exception as e:
                print(f"❌ Failed to optimize {method}: {e}")
                continue
        else:
            print(f"⚠️  Unknown method: {method}")

    return results