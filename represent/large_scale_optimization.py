"""
Large-Scale Parameter Optimization for Symbol Datasets

This module provides efficient parameter optimization for datasets with millions of samples
using intelligent sampling strategies and parallel evaluation.
"""

import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Try to import Optuna for optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

OPTIMIZATION_AVAILABLE = OPTUNA_AVAILABLE

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm  # noqa: F401
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

from .target_generators.base import TargetGenerator  # noqa: E402


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
        # Adaptive sampling parameters
        adaptive_sampling: bool = True,
        min_window_size: int = 15000,  # USER REQ: Reduced to 15K min sampling window
        max_window_size: int = 25000,  # USER REQ: Reduced to 25K max sampling window
        stabilization_threshold: float = 0.05,
        stabilization_patience: int = 3,
        growth_factor: float = 1.5,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        # Class balance penalty
        class_balance_weight: float = 0.5,  # Weight for class balance penalty
        # Robust sampling parameters
        target_coverage: float = 0.25,  # Target 25% dataset coverage
        use_cross_validation: bool = True,  # Enable cross-validation
        validation_split: float = 0.3,  # 30% for validation
        # Debug logging
        debug_log_path: str | Path | None = None,
        # Sequential processing (to match application behavior)
        use_sequential_processing: bool = False,  # NEW: Process full dataset sequentially
        sequential_subset_size: int | None = None,  # Optional subset size for large datasets
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
            use_sequential_processing: Process full dataset sequentially to match application (default: False)
        """
        if not OPTIMIZATION_AVAILABLE:
            raise ImportError(
                "Parameter optimization requires Optuna. "
                "Install with: pip install optuna"
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

        # Optuna is the only backend
        self.use_optuna = True

        # Adaptive sampling parameters
        self.adaptive_sampling = adaptive_sampling
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.stabilization_threshold = stabilization_threshold
        self.stabilization_patience = stabilization_patience
        self.growth_factor = growth_factor
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.class_balance_weight = class_balance_weight
        # Robust sampling parameters
        self.target_coverage = target_coverage
        self.use_cross_validation = use_cross_validation
        self.validation_split = validation_split
        # Debug logging path
        self.debug_log_path = Path(debug_log_path) if debug_log_path else None
        # Sequential processing flag
        self.use_sequential_processing = use_sequential_processing
        self.sequential_subset_size = sequential_subset_size

        # Adaptive sampling state
        self.current_window_size = window_size
        self.current_n_windows = n_windows
        self.parameter_history: list[dict[str, int | float]] = []
        self.stable_count = 0

        # Setup random state
        self.rng = np.random.RandomState(random_state)

        # Progress reporting counters
        self._sampling_call_count = 0

    def _run_optuna_optimization(self, original_objective, bounds: dict, method_name: str,
                                generator_class, prices):
        """Run optimization using Optuna with TPE sampler."""
        if self.verbose:
            print("   🎯 Using Optuna TPE optimizer")

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
                if param_name in ['population_size', 'max_generations', 'lookforward_window', 'lookback_window',
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
                    'population_size', 'max_generations', 'lookforward_window', 'lookback_window', 'min_trades',
                    'window_size', 'volatility_window'
                }

                casted_params = {}
                for k, v in filtered_params.items():
                    if k in int_like_params and isinstance(v, int | float):
                        casted_params[k] = int(v)
                    else:
                        casted_params[k] = v

                # Sample multiple windows and evaluate
                if self.use_sequential_processing:
                    # Use dataset sequentially to match application behavior
                    if self.sequential_subset_size and len(prices) > self.sequential_subset_size:
                        # Use random subset for large datasets
                        subset_indices = self.rng.choice(len(prices), self.sequential_subset_size, replace=False)
                        subset_indices.sort()  # Keep temporal order
                        subset_prices = prices[subset_indices]
                        windows = [subset_prices]
                        current_n_windows = 1
                    else:
                        # Use full dataset
                        windows = [prices]
                        current_n_windows = 1
                else:
                    # Standard windowed sampling
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

                        elif method_name in ['Triple Barrier Adaptive (Large-Scale)']:
                            # Use pre-calculated returns from the adaptive method
                            returns = targets[f"{generator.target_name}_return"].to_numpy()
                            window_pnl = returns.sum()  # Returns already include transaction costs
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

                # Calculate class balance score as primary optimization objective
                balance_score = 0.0
                if method_name in ['Triple Barrier (Large-Scale)', 'Triple Barrier Adaptive (Large-Scale)', 'Triple Exceedance (Large-Scale)']:
                    all_labels = []
                    for window_prices in windows:
                        try:
                            generator = generator_class(**casted_params)
                            window_df = pl.DataFrame({
                                "ts_event": range(len(window_prices)),
                                "mid_price": window_prices,
                            })
                            targets = generator.generate_targets(window_df)
                            target_info = generator.get_target_info()
                            target_col = target_info['target_names'][0]
                            window_labels = targets[target_col].to_numpy()
                            all_labels.extend(window_labels.tolist())
                        except Exception:
                            continue

                    if len(all_labels) > 0:
                        # Calculate class balance score (min/max * 100, higher is better)
                        unique_labels, counts = np.unique(all_labels, return_counts=True)
                        if len(unique_labels) > 1:
                            percentages = counts / len(all_labels) * 100
                            balance_score = min(percentages) / max(percentages) * 100
                        else:
                            balance_score = 100.0  # Perfect balance for single class

                        if self.verbose:
                            print(f"   ⚖️  Class balance score: {balance_score:.1f}% (higher is better)")
                            # Convert to integer percentages for cleaner display
                            int_percentages = [f"{p:.0f}%" for p in percentages]
                            print(f"      Distribution: {dict(zip(unique_labels, int_percentages, strict=True))}")
                            print(f"      Returns: {avg_return:.3f}, Balance: {balance_score:.1f}%")

                # Track parameter history for stability analysis (if adaptive sampling is enabled)
                if self.adaptive_sampling:
                    self.parameter_history.append(params.copy())

                    # Check parameter stability after initial phase
                    if evaluation_count[0] > self.initial_points:
                        self._check_parameter_stability(params)

                        # Now check if we should increase sampling based on updated stability
                        if self._should_increase_sampling():
                            self._increase_sampling()

                # Optimize ONLY for class balance score (minimize negative balance score)
                current_return = -balance_score  # Minimize negative balance score
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
            print("\\n🎯 Optimization Summary:")
            print(f"   Trials completed: {len(study.trials)}")
            # Display return in percentage format - our returns are already properly scaled
            if result_fun is not None:
                return_pct = -result_fun * 100
                print(f"   Best return: {-result_fun:.4f} ({return_pct:.2f}%)")
            else:
                print("   Best return: No valid result found")
            print(f"   Best params: {best_trial.params}")

        # Create a simple result object that mimics scikit-optimize result
        class OptunaResult:
            def __init__(self, x, fun, trials, best_params):
                self.x = x
                self.fun = fun
                self.best_params = best_params
                self.func_vals = [t.value for t in trials if t.value is not None]
                self.x_iters = [[t.params[name] for name in bounds.keys()] for t in trials if t.value is not None]

            def get(self, key, default=None):
                return getattr(self, key, default)

            def __getitem__(self, key):
                return getattr(self, key)

            def __contains__(self, key):
                return hasattr(self, key)

        return OptunaResult(result_x, result_fun, study.trials, best_trial.params if best_trial else {})

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
                print("   📊 Stabilized parameters:")
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

        # Calculate coverage for debug logging
        total_samples_used = sum(len(w) for w in windows)
        coverage = (total_samples_used / total_samples) * 100

        if self.verbose:
            self._sampling_call_count += 1
            # Don't print sampling info during optimization when TQDM progress bar is active
            # This prevents interference with inline progress updates
            # Only print sampling info every 10th call when not using TQDM
            if not hasattr(self, '_tqdm_active') or not self._tqdm_active:
                if self._sampling_call_count % 10 == 1:
                    print(f"   📊 Sampled {len(windows)} windows ({total_samples_used:,} samples, {coverage:.1f}% coverage)")
        else:
            self._sampling_call_count += 1
            # Debug log sampled window indices if logging is enabled
            if self.debug_log_path:
                self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.debug_log_path, 'a') as f:
                    f.write(f"SAMPLED_WINDOWS total={len(windows)} coverage={coverage:.3f}% size={window_size}\n")

        return windows

    def sample_windows_robust(
        self,
        prices: np.ndarray,
        n_windows: int | None = None,
        target_coverage: float = 0.25,
        validation_split: float = 0.3
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Sample windows with robust cross-validation and higher coverage.

        This addresses the overfitting issue where optimization uses only 2.3%
        of the dataset by increasing coverage to 20-30% and using cross-validation.

        Args:
            prices: Full price series
            n_windows: Number of windows (uses self.n_windows if None)
            target_coverage: Target dataset coverage (0.25 = 25%)
            validation_split: Fraction of windows for validation (0.3 = 30%)

        Returns:
            Tuple of (training_windows, validation_windows)
        """
        if n_windows is None:
            n_windows = self.n_windows

        total_samples = len(prices)
        window_size = self.current_window_size if self.adaptive_sampling else self.window_size

        # Calculate number of windows needed for target coverage
        samples_per_window = window_size
        target_samples = int(total_samples * target_coverage)
        required_windows = max(n_windows, target_samples // samples_per_window)

        if self.verbose:
            current_coverage = (n_windows * window_size / total_samples) * 100
            new_coverage = (required_windows * window_size / total_samples) * 100
            print(f"   🔄 Robust sampling: {current_coverage:.1f}% → {new_coverage:.1f}% coverage")
            print(f"   📊 Windows: {n_windows} → {required_windows} ({validation_split:.0%} for validation)")

        # Generate more windows with stratified sampling for better representation
        all_windows = []

        if self.sampling_strategy == "stratified":
            # Create stratified segments across the entire time series
            segment_size = total_samples // required_windows

            for i in range(required_windows):
                segment_start = i * segment_size
                segment_end = min((i + 1) * segment_size, total_samples)

                if segment_end - segment_start >= window_size:
                    # Randomly sample within each segment for diversity
                    max_start = segment_end - window_size
                    start_idx = self.rng.randint(segment_start, max_start + 1)
                    end_idx = start_idx + window_size
                    all_windows.append(prices[start_idx:end_idx])

        elif self.sampling_strategy == "uniform":
            # Uniform random sampling with higher coverage
            for _ in range(required_windows):
                start_idx = self.rng.randint(0, total_samples - window_size + 1)
                end_idx = start_idx + window_size
                all_windows.append(prices[start_idx:end_idx])

        else:
            # Fallback to temporal sampling
            step_size = (total_samples - window_size) // (required_windows - 1) if required_windows > 1 else 0
            for i in range(required_windows):
                start_idx = min(i * step_size, total_samples - window_size)
                end_idx = start_idx + window_size
                all_windows.append(prices[start_idx:end_idx])

        # Split into training and validation sets
        n_validation = max(1, int(len(all_windows) * validation_split))
        n_training = len(all_windows) - n_validation

        # Shuffle and split
        indices = np.arange(len(all_windows))
        self.rng.shuffle(indices)

        train_indices = indices[:n_training]
        val_indices = indices[n_training:]

        training_windows = [all_windows[i] for i in train_indices]
        validation_windows = [all_windows[i] for i in val_indices]

        # Report coverage
        total_samples_used = sum(len(w) for w in all_windows)
        coverage = (total_samples_used / total_samples) * 100

        if self.verbose:
            print(f"   ✅ Robust sampling complete: {coverage:.1f}% coverage, {n_training} train + {n_validation} val windows")

        return training_windows, validation_windows

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
        if isinstance(prices, str | Path):
            if data_loader is None:
                raise ValueError("data_loader function required when prices is a file path")
            prices = data_loader(prices)

        # Type assertion to help type checker understand prices is now ndarray
        assert isinstance(prices, np.ndarray), "prices must be ndarray after conversion"

        total_samples = len(prices)

        if self.verbose:
            print(f"🔍 Large-scale optimization: {method_name}")
            print(f"   📈 Dataset: {total_samples:,} samples ({total_samples/1e6:.1f}M)")
            print(f"   📊 Sampling: {self.current_window_size:,} samples × {self.current_n_windows} windows per trial")
            if bounds:
                print(f"   🔍 Using Optuna TPE optimizer ({self.n_calls} trials)")
                print(f"   🎯 Bounds: {bounds}")
            else:
                print("   🎯 Fixed parameters method (single evaluation with 0.7 pip transaction cost)")

        # Handle methods with no parameters to optimize (e.g., Oracle with fixed transaction costs)
        if not bounds:
            if self.verbose:
                print(f"🎯 Evaluating {method_name} with fixed parameters...")

            # Single evaluation with fixed transaction cost
            try:
                # Create generator with fixed 0.35 pip one-way transaction cost (0.7 pip round-trip)
                generator_params = {'transaction_cost': 0.000035}  # Fixed at 0.35 pips (one-way)

                # Filter out incompatible params for generators that don't use transaction_cost
                filtered_params = generator_params.copy()
                if 'CTL' in generator_class.__name__ and 'transaction_cost' in filtered_params:
                    filtered_params.pop('transaction_cost')

                generator = generator_class(**filtered_params)

                # Sample windows for evaluation
                if self.use_sequential_processing:
                    # Use dataset sequentially to match application behavior
                    if self.sequential_subset_size and len(prices) > self.sequential_subset_size:
                        # Use random subset for large datasets
                        subset_indices = self.rng.choice(len(prices), self.sequential_subset_size, replace=False)
                        subset_indices.sort()  # Keep temporal order
                        subset_prices = prices[subset_indices]
                        windows = [subset_prices]
                    else:
                        # Use full dataset
                        windows = [prices]
                else:
                    # Standard windowed sampling
                    windows = self.sample_windows(prices, self.current_n_windows)

                # Evaluate across all windows
                total_pnl = 0.0
                valid_evaluations = 0

                for window in windows:
                    window_prices = window
                    if len(window_prices) < 100:  # Skip very small windows
                        continue

                    # Create DataFrame for generator
                    window_df = pl.DataFrame({"mid_price": window_prices})
                    targets_df = generator.generate_targets(window_df)
                    target_info = generator.get_target_info()
                    labels = targets_df[target_info['target_names'][0]].to_numpy()
                    normalized_labels = self._normalize_labels_for_pnl(labels)

                    # Apply transaction costs (already halved for round-trip)
                    half_fee = (self.fee_pips * 0.0001) / 2.0
                    # Define local helper function for PnL calculation
                    def _estimate_directional_pnl_local(prices_arr: np.ndarray, labels_arr: np.ndarray, fee: float) -> float:
                        pnl = 0.0
                        position = 0  # 0: flat, 1: long, -1: short
                        entry_price = 0.0

                        for i in range(len(labels_arr)):
                            label = labels_arr[i]
                            price = prices_arr[i]

                            if label == 1 and position != 1:  # Go long
                                if position == -1:  # Close short first
                                    pnl += (entry_price - price) - fee
                                entry_price = price
                                position = 1
                                pnl -= fee  # Entry cost
                            elif label == -1 and position != -1:  # Go short
                                if position == 1:  # Close long first
                                    pnl += (price - entry_price) - fee
                                entry_price = price
                                position = -1
                                pnl -= fee  # Entry cost
                            elif label == 0 and position != 0:  # Close position
                                if position == 1:
                                    pnl += (price - entry_price) - fee
                                elif position == -1:
                                    pnl += (entry_price - price) - fee
                                position = 0

                        # Close any remaining position
                        if position != 0:
                            final_price = prices_arr[-1]
                            if position == 1:
                                pnl += (final_price - entry_price) - fee
                            elif position == -1:
                                pnl += (entry_price - final_price) - fee

                        return pnl

                    window_pnl = _estimate_directional_pnl_local(window_prices, normalized_labels, half_fee)

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

        # Use Optuna-only optimization
        # Initialize tracking variables
        evaluation_count = [0]
        best_return_so_far = [float('-inf')]
        progress_bar = None
        result_x = {}  # Will store best parameters

        def objective(trial):
            """Optuna objective function using adaptive sampled windows."""
            nonlocal evaluation_count, best_return_so_far, progress_bar, result_x
            evaluation_count[0] += 1

            # Sample parameters from bounds
            params = {}
            for param_name, (low, high) in bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)

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
                    'population_size', 'max_generations', 'lookforward_window', 'lookback_window', 'min_trades',
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

                # Choose evaluation strategy based on configuration
                if self.use_sequential_processing:
                    # NEW: Process dataset sequentially to match application behavior
                    if self.sequential_subset_size and len(prices) > self.sequential_subset_size:
                        # Use random subset for large datasets
                        subset_indices = self.rng.choice(len(prices), self.sequential_subset_size, replace=False)
                        subset_indices.sort()  # Keep temporal order
                        subset_prices = prices[subset_indices]
                        windows = [subset_prices]
                        if self.verbose:
                            print(f"      🔄 Using sequential processing on subset ({len(subset_prices):,} / {len(prices):,} samples)")
                    else:
                        # Use full dataset
                        windows = [prices]
                        if self.verbose:
                            print(f"      🔄 Using sequential processing on full dataset ({len(prices):,} samples)")
                    cv_windows = []
                elif self.use_cross_validation:
                    # Use robust sampling with cross-validation
                    training_windows, validation_windows = self.sample_windows_robust(
                        prices,
                        n_windows=self.current_n_windows,
                        target_coverage=self.target_coverage,
                        validation_split=self.validation_split
                    )
                    # Use training windows for optimization, validation for final check
                    windows = training_windows
                    cv_windows = validation_windows
                    if self.verbose:
                        print(f"      🔄 Using cross-validated sampling ({len(windows)} training windows)")
                else:
                    # Standard sampling for backwards compatibility
                    windows = self.sample_windows(prices, n_windows=self.current_n_windows)
                    cv_windows = []
                    if self.verbose:
                        print(f"      🔄 Using standard windowed sampling ({len(windows)} windows)")

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

                # Calculate class balance score as primary optimization objective
                balance_score = 0.0
                if method_name in ['Triple Barrier (Large-Scale)', 'Triple Barrier Adaptive (Large-Scale)', 'Triple Exceedance (Large-Scale)']:
                    all_labels = []
                    # Use the same windows as the main evaluation (already handles sequential vs windowed)
                    evaluation_windows = windows

                    for window_prices in evaluation_windows:
                        try:
                            generator = generator_class(**casted_params)
                            window_df = pl.DataFrame({
                                "ts_event": range(len(window_prices)),
                                "mid_price": window_prices,
                            })
                            targets = generator.generate_targets(window_df)
                            target_info = generator.get_target_info()
                            target_col = target_info['target_names'][0]
                            window_labels = targets[target_col].to_numpy()
                            all_labels.extend(window_labels.tolist())
                        except Exception:
                            continue

                    if len(all_labels) > 0:
                        # Calculate class balance score (min/max * 100, higher is better)
                        unique_labels, counts = np.unique(all_labels, return_counts=True)
                        if len(unique_labels) > 1:
                            percentages = counts / len(all_labels) * 100
                            balance_score = min(percentages) / max(percentages) * 100
                        else:
                            balance_score = 100.0  # Perfect balance for single class

                        if self.verbose and evaluation_count[0] <= 3:
                            print(f"   ⚖️  Class balance score: {balance_score:.1f}% (higher is better)")
                            # Convert to integer percentages for cleaner display
                            int_percentages = [f"{p:.0f}%" for p in percentages]
                            print(f"      Distribution: {dict(zip(unique_labels, int_percentages, strict=True))}")
                            print(f"      Returns: {avg_returns:.3f}, Balance: {balance_score:.1f}%")

                # Cross-validation check if enabled
                cv_penalty = 0.0
                if self.use_cross_validation and cv_windows:
                    cv_balance_score = 0.0
                    cv_all_labels = []

                    # Evaluate on validation windows
                    for cv_window_prices in cv_windows:
                        try:
                            cv_df = pl.DataFrame({
                                'mid_price': cv_window_prices,
                                'timestamp': range(len(cv_window_prices))
                            })

                            cv_result_df = generator.generate_targets(cv_df)
                            if hasattr(generator, 'target_name'):
                                target_name = generator.target_name  # type: ignore[attr-defined]
                                cv_labels = cv_result_df[target_name].to_numpy()
                            else:
                                # Fallback for generators without target_name
                                cv_cols = [col for col in cv_result_df.columns if 'label' in col.lower()]
                                if cv_cols:
                                    cv_labels = cv_result_df[cv_cols[0]].to_numpy()
                                else:
                                    continue

                            cv_all_labels.extend(cv_labels)
                        except Exception:
                            continue

                    # Calculate validation balance score
                    if cv_all_labels:
                        cv_unique_labels, cv_counts = np.unique(cv_all_labels, return_counts=True)
                        if len(cv_unique_labels) > 1:
                            cv_percentages = (cv_counts / len(cv_all_labels)) * 100
                            cv_balance_score = min(cv_percentages) / max(cv_percentages) * 100
                        else:
                            cv_balance_score = 100.0

                        # Penalize if validation performance is much worse than training
                        performance_gap = balance_score - cv_balance_score
                        if performance_gap > 10.0:  # More than 10% gap suggests overfitting
                            cv_penalty = performance_gap * 0.5  # Moderate penalty

                        if self.verbose and evaluation_count[0] <= 3:
                            print(f"   🔄 CV balance score: {cv_balance_score:.1f}% (gap: {performance_gap:.1f}%, penalty: {cv_penalty:.1f})")

                # Final score with cross-validation penalty
                final_score = balance_score - cv_penalty

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
                    best_return_so_far[0] = float(final_score)
                    result_x = params.copy()  # Store best parameters

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
                # Use Optuna for optimization
                result = self._run_optuna_optimization(
                    None, bounds, method_name, generator_class, prices
                )
            except EarlyStoppingException:
                # Early stopping was triggered - create a partial result
                if self.verbose:
                    print(f"   ✅ Early stopping completed after {evaluation_count[0]} evaluations")

                # Create a mock result object with the best parameters found so far
                # We'll use the last parameters from history as the "best" result
                if self.parameter_history:
                    best_params_dict = self.parameter_history[-1]
                    result_fun = -float('inf')  # No meaningful result for broken code

                    # Create a simple result object
                    class EarlyStopResult:
                        def __init__(self, x, fun, best_params):
                            self.x = x
                            self.fun = -fun  # Convert back to minimization format
                            self.func_vals = []  # Empty list for compatibility
                            self.x_iters = []    # Empty list for compatibility
                            self.best_params = best_params  # Store best parameters

                        def get(self, key, default=None):
                            """Provide dict-like get method for compatibility."""
                            return getattr(self, key, default)

                        def __getitem__(self, key):
                            """Allow dict-like access."""
                            return getattr(self, key)

                        def __contains__(self, key):
                            """Allow 'in' operator."""
                            return hasattr(self, key)

                    result = EarlyStopResult(result_x, result_fun, best_params_dict)
                else:
                    # Fallback if no history exists
                    raise RuntimeError("Early stopping triggered but no parameter history available") from None  # noqa: B904

            # Finish progress display and cleanup flag
            if self.verbose:
                print()  # New line after progress display
                self._tqdm_active = False  # Re-enable sampling messages

        # Extract optimal parameters
        optimal_params: dict[str, int | float] = {}
        # Type assertion to help PyRight understand result is not None
        # Process Optuna results (scikit-optimize result processing removed)
        if hasattr(result, 'best_params') and result.best_params:
            optimal_params = result.best_params.copy()
            # Ensure integer parameters are properly cast
            for param_name in ['population_size', 'max_generations', 'lookforward_window',
                              'lookback_window', 'min_trades', 'window_size', 'volatility_window']:
                if param_name in optimal_params:
                    optimal_params[param_name] = int(optimal_params[param_name])

        optimal_returns = -result.fun if result.fun is not None else 0.0

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

        # FINAL EVALUATION: Apply best parameters to entire dataset
        if self.verbose:
            print("\n🔄 Final evaluation on entire dataset...")

        try:
            # Create generator with optimal parameters
            final_generator = generator_class(**optimal_params)

            # Apply to full dataset
            if self.use_sequential_processing:
                # Use full dataset for final evaluation
                final_data = prices
            else:
                # Use a large representative sample for final evaluation
                max_final_samples = min(1_000_000, len(prices))  # Up to 1M samples
                if len(prices) > max_final_samples:
                    final_indices = self.rng.choice(len(prices), max_final_samples, replace=False)
                    final_indices.sort()  # Keep temporal order
                    final_data = prices[final_indices]
                else:
                    final_data = prices

            # Create DataFrame and generate targets
            final_df = pl.DataFrame({
                'mid_price': final_data,
                'ts_event': range(len(final_data))
            })

            final_targets = final_generator.generate_targets(final_df)
            target_info = final_generator.get_target_info()
            final_labels = final_targets[target_info['target_names'][0]].to_numpy()

            # Calculate final metrics
            unique_labels, counts = np.unique(final_labels[~np.isnan(final_labels)], return_counts=True)
            total_final = len(final_labels[~np.isnan(final_labels)])

            if len(unique_labels) > 1:
                percentages = counts / total_final * 100
                balance_score = min(percentages) / max(percentages) * 100
                int_percentages = [f"{p:.0f}%" for p in percentages]

                if self.verbose:
                    print(f"📊 FINAL DATASET METRICS ({len(final_data):,} samples):")
                    print(f"   ⚖️  Class balance: {balance_score:.1f}% ({len(unique_labels)} classes)")
                    print(f"   📈 Label distribution: {dict(zip(unique_labels, int_percentages, strict=True))}")

                    # Signal frequency
                    non_zero_signals = np.count_nonzero(final_labels[~np.isnan(final_labels)])
                    signal_freq = non_zero_signals / total_final * 100
                    print(f"   📊 Signal frequency: {signal_freq:.1f}% ({non_zero_signals:,}/{total_final:,})")

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Final evaluation failed: {e}")

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
            raise ImportError("CTL labeling requires tstrends integration") from None  # noqa: B904

        # FIXED CTL PARAMETERS: prevent zero omega that causes position changes on every tick
        # Minimum omega set to 0.0001 (1 pip) to ensure reasonable threshold for trend detection
        default_bounds = {'omega': (0.0001, 0.001)}  # Realistic range prevents pathological overtrading
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
            raise ImportError("CTL labeling requires tstrends integration") from None  # noqa: B904

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
            raise ImportError("Oracle labeling requires tstrends integration") from None  # noqa: B904

        # FIXED Oracle Binary: use realistic 0.35 pip one-way cost (0.7 pip round-trip)
        # Transaction cost should be fixed, not optimized, for realistic trading
        default_bounds = {}
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
            raise ImportError("Oracle labeling requires tstrends integration") from None  # noqa: B904

        # FIXED Oracle Ternary: use realistic 0.35 pip cost, optimize neutral reward factor
        # Transaction cost fixed at 0.35 pips (0.7 pip round-trip), only optimize neutral factor
        default_bounds = {
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
            'lookforward_window': (1000, 10000),    # 1K-3K ticks: proven range around 2K
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

    def optimize_triple_barrier_adaptive(
        self,
        prices: np.ndarray | str | Path,
        custom_bounds: dict[str, tuple] | None = None,
        data_loader: Callable | None = None
    ) -> dict[str, Any]:
        """Optimize Triple Barrier with window sampling."""
        from .target_generators.triple_barrier_adaptive import TripleBarrierGeneratorAdaptive

        # PROVEN bounds based on successful diagnostic parameters (2K window, 1 pip barriers)
        # Lookforward limited to 5K samples max, sampling windows to 25K max
        # Bounds centered around proven diagnostic values that generated good signals
        default_bounds = {
            'lookforward_window': (1000, 5000),    # 1K-5K ticks: more reasonable range
            'barrier_width': (0.5, 2.0), # 0.5-2 sigma barriers: tighter for better signal generation
            'lookback_window': (1000, 5000),    # 1K-5K ticks: more reasonable range
        }

        if custom_bounds:
            default_bounds.update(custom_bounds)

        return self.optimize_with_sampling(
            TripleBarrierGeneratorAdaptive,
            prices,
            default_bounds,
            "Triple Barrier Adaptive (Large-Scale)",
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
