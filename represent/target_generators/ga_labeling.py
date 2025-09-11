"""
Genetic Algorithm (GA) Labeling Target Generator

This module implements a genetic algorithm approach for generating trading labels
based on the methodology from https://github.com/SCH-YcHan/GA-labeling.

The GA labeling approach uses genetic algorithms to evolve optimal trading signal
patterns by maximizing trading performance metrics like win rate, profit factor,
and total profitability.

Key Features:
- Genetic algorithm optimization for trading labels
- Trading performance-based fitness evaluation
- Binary chromosome encoding for trading signals
- Configurable population size and evolution parameters
- Transaction cost consideration in fitness calculation

References:
- Original GA labeling repository: https://github.com/SCH-YcHan/GA-labeling
- Genetic algorithms for financial time series labeling
"""

import warnings
from typing import Any

import numpy as np
import polars as pl

from .base import TargetGenerator


class GALabelingGenerator(TargetGenerator):
    """
    Genetic Algorithm-based trading label generator with dual long/short models.

    This generator creates TWO separate GA-optimized models:
    1. Long Model: Optimized for long positions (BUY/HOLD signals)
    2. Short Model: Optimized for short positions (SELL/HOLD signals)
    
    Each model is independently evolved to maximize performance for its specific
    trading direction, resulting in specialized long and short trading strategies.
    
    Outputs:
    - long_labels: Binary labels (1: BUY long, 0: HOLD)
    - short_labels: Binary labels (1: SELL short, 0: HOLD)
    """

    @property
    def required_columns(self) -> list[str]:
        """Return list of required DataFrame columns."""
        return ["mid_price"]

    @property
    def target_type(self) -> str:
        """Return the type of targets generated."""
        return "classification"

    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        lookforward_window: int = 100,  # Hundreds of ticks for meaningful predictions
        transaction_cost: float = 0.00007,  # 0.7 pips = 0.00007 (0.007%)
        min_trades: int = 30,
        max_trade_frequency: float = 0.05,  # Max 5% of samples can be trades
        min_win_rate: float = 0.33,
        max_win_rate: float = 0.80,
        min_profit_factor: float = 1.0,
        elitism: int = 3,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.8,
        target_name_prefix: str = "ga",  # Prefix for long/short columns
        random_seed: int | None = None,
        verbose: bool = False,
        chunk_size: int = 10000,  # Process population in chunks to manage memory
        dual_models: bool = True  # Generate separate long and short models
    ):
        """
        Initialize GA labeling generator.

        Args:
            population_size: Size of GA population (default: 50)
            max_generations: Maximum number of GA generations (default: 100)
            lookforward_window: Ticks ahead to predict (default: 100)
            transaction_cost: Trading transaction cost as decimal (default: 0.00007 = 0.7 pips)
            min_trades: Minimum trades required for valid strategy (default: 30)
            min_win_rate: Minimum win rate threshold (default: 0.33)
            max_win_rate: Maximum win rate threshold (default: 0.80)
            min_profit_factor: Minimum profit factor required (default: 1.0)
            elitism: Number of best solutions to preserve (default: 3)
            mutation_rate: Probability of mutation (default: 0.01)
            crossover_rate: Probability of crossover (default: 0.8)
            target_name_prefix: Prefix for target columns (default: "ga")
            random_seed: Random seed for reproducibility (default: None)
            verbose: Whether to print optimization progress (default: False)
            chunk_size: Process chromosomes in chunks to manage memory (default: 10000)
            dual_models: Generate separate long and short models (default: True)
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.lookforward_window = lookforward_window
        self.transaction_cost = transaction_cost
        self.min_trades = min_trades
        self.max_trade_frequency = max_trade_frequency
        self.min_win_rate = min_win_rate
        self.max_win_rate = max_win_rate
        self.min_profit_factor = min_profit_factor
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.target_name_prefix = target_name_prefix
        self.random_seed = random_seed
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.dual_models = dual_models

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_targets(self, df: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
        """Generate GA-optimized trading labels for both long and short models."""
        self.validate_input(df)

        # Extract price data
        prices = df["mid_price"].to_numpy()

        if len(prices) < self.min_trades * 2:
            warnings.warn(
                f"Insufficient data for GA labeling: {len(prices)} samples. "
                f"Need at least {self.min_trades * 2} samples. Returning zero labels.",
                stacklevel=2
            )
            long_labels = np.zeros(len(prices), dtype=np.int32)
            short_labels = np.zeros(len(prices), dtype=np.int32)
        else:
            if self.dual_models:
                # Generate separate long and short models
                long_labels = self._optimize_long_model(prices)
                short_labels = self._optimize_short_model(prices)
            else:
                # Legacy single model mode
                labels = self._optimize_labels(prices)
                long_labels = labels
                short_labels = np.zeros(len(prices), dtype=np.int32)  # No short model

        # Create base target DataFrame with keys
        target_df = self._create_base_target_df(df, symbol)

        # Add both target columns
        target_df = target_df.with_columns([
            pl.Series(f"{self.target_name_prefix}_long_labels", long_labels),
            pl.Series(f"{self.target_name_prefix}_short_labels", short_labels)
        ])

        return target_df

    def _optimize_long_model(self, prices: np.ndarray) -> np.ndarray:
        """
        Optimize GA model specifically for long positions.
        
        Args:
            prices: Array of prices to optimize labels for
            
        Returns:
            Binary labels optimized for long trading (1: BUY, 0: HOLD)
        """
        if self.verbose:
            print(f"🟢 Optimizing LONG model...")
        
        return self._run_ga_optimization(prices, model_type="long")
    
    def _optimize_short_model(self, prices: np.ndarray) -> np.ndarray:
        """
        Optimize GA model specifically for short positions.
        
        Args:
            prices: Array of prices to optimize labels for
            
        Returns:
            Binary labels optimized for short trading (1: SELL, 0: HOLD)
        """
        if self.verbose:
            print(f"🔴 Optimizing SHORT model...")
            
        return self._run_ga_optimization(prices, model_type="short")

    def _optimize_labels(self, prices: np.ndarray) -> np.ndarray:
        """
        Legacy method for single model optimization.

        Args:
            prices: Array of prices to optimize labels for

        Returns:
            Optimized binary labels (0: hold, 1: buy)
        """
        return self._run_ga_optimization(prices, model_type="long")
        
    def _run_ga_optimization(self, prices: np.ndarray, model_type: str = "long") -> np.ndarray:
        """
        Run genetic algorithm to optimize trading labels for specific model type.

        Args:
            prices: Array of prices to optimize labels for
            model_type: "long" for long model, "short" for short model

        Returns:
            Optimized binary labels (1: trade signal, 0: hold)
        """
        n_samples = len(prices)

        if self.verbose:
            direction = "LONG" if model_type == "long" else "SHORT"
            print(f"🧬 Running {direction} GA optimization for {n_samples} samples")
            print(f"   Population: {self.population_size}, Generations: {self.max_generations}")

        # Initialize population
        population = self._initialize_population(n_samples)

        best_fitness = float('-inf')
        best_chromosome = None
        generation_without_improvement = 0

        for generation in range(self.max_generations):
            # Evaluate fitness for all chromosomes in chunks to manage memory
            fitness_scores = []
            for i in range(0, len(population), min(self.chunk_size, len(population))):
                chunk = population[i:i + min(self.chunk_size, len(population))]
                chunk_fitness = [
                    self._evaluate_fitness(chromosome, prices, model_type)
                    for chromosome in chunk
                ]
                fitness_scores.extend(chunk_fitness)
            
            fitness_scores = np.array(fitness_scores)

            # Track best solution
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[current_best_idx].copy()
                generation_without_improvement = 0
            else:
                generation_without_improvement += 1

            if self.verbose and generation % 100 == 0:
                print(f"   Generation {generation}: Best fitness = {best_fitness:.4f}")

            # Early stopping if no improvement for many generations
            if generation_without_improvement > 100:
                if self.verbose:
                    print(f"   Early stopping at generation {generation}")
                break

            # Create next generation
            population = self._create_next_generation(population, fitness_scores)

        if self.verbose:
            print(f"   🎯 Final best fitness: {best_fitness:.4f}")

        # Always return the best evolved chromosome - never fallback to simple labeling
        if best_chromosome is not None:
            return best_chromosome.astype(np.int32)
        else:
            # If no best chromosome found, return first chromosome from final population
            # This ensures we always use GA-evolved strategy, never simple directional labeling
            if len(population) > 0:
                return population[0].astype(np.int32)
            else:
                # Emergency fallback: return random binary labels (should never happen)
                return np.random.randint(0, 2, size=len(prices), dtype=np.int32)

    def _initialize_population(self, n_samples: int) -> list[np.ndarray]:
        """
        Initialize population with sparse (low-frequency) binary chromosomes.
        Start with chromosomes that respect the max_trade_frequency constraint.

        Args:
            n_samples: Number of samples (chromosome length)

        Returns:
            List of individual chromosomes to avoid large matrix allocation
        """
        population = []
        for i in range(self.population_size):
            # Create sparse chromosomes that start within trade frequency limits
            # Use probability based on max_trade_frequency to avoid starting with overtrading chromosomes
            trade_probability = min(self.max_trade_frequency * 0.8, 0.1)  # Start conservatively
            
            # Add some variation across population  
            if i < self.population_size // 4:
                # 25% very conservative 
                trade_prob = trade_probability * 0.5
            elif i < self.population_size // 2:
                # 25% at target
                trade_prob = trade_probability
            else:
                # 50% slightly above target (but still reasonable)
                trade_prob = min(trade_probability * 1.5, 0.15)
                
            chromosome = np.random.choice([0, 1], size=n_samples, 
                                        p=[1-trade_prob, trade_prob]).astype(np.int8)
            population.append(chromosome)
        return population

    def _evaluate_fitness(self, chromosome: np.ndarray, prices: np.ndarray, model_type: str = "long") -> float:
        """
        Evaluate fitness of a chromosome based on trading performance.

        Args:
            chromosome: Binary array representing trading decisions (1: trade, 0: hold)
            prices: Price array for simulation
            model_type: "long" for long model, "short" for short model

        Returns:
            Fitness score (higher is better)
        """
        # Simulate trading based on chromosome and model type
        trades = self._simulate_specialized_trading(chromosome, prices, model_type)

        if len(trades) == 0:
            return -1000.0  # Penalty for no trades (but chromosome still valid)

        # Calculate trading metrics
        n_trades = len(trades)
        wins = [trade for trade in trades if trade > 0]
        losses = [trade for trade in trades if trade <= 0]

        # Check minimum trade requirements
        if n_trades < self.min_trades:
            return -500.0  # Penalty for insufficient trades (but chromosome still valid)

        win_rate = len(wins) / n_trades if n_trades > 0 else 0

        # Check win rate constraints
        if win_rate < self.min_win_rate or win_rate > self.max_win_rate:
            return -200.0  # Penalty for win rate outside acceptable range (but chromosome still valid)

        # Calculate profit metrics
        mean_win = np.mean(wins) if wins else 0
        mean_loss = abs(np.mean(losses)) if losses else 1  # Avoid division by zero

        profit_factor = mean_win / mean_loss if mean_loss > 0 else 0

        # Check profit factor requirement
        if profit_factor < self.min_profit_factor:
            return -100.0  # Penalty for insufficient profit factor (but chromosome still valid)

        # Calculate fitness score prioritizing net return per trade while heavily penalizing overtrading
        total_return = np.sum(trades)
        
        # STRONG penalty for excessive trading - transaction costs kill profits
        # Use configurable max trade frequency (default 5% of samples)
        max_reasonable_trades = max(len(prices) * self.max_trade_frequency, self.min_trades)  # At least min_trades allowed
        
        if n_trades > max_reasonable_trades:
            # EXTREME penalty for overtrading - must dominate fitness to force compliance
            excess_trades = n_trades - max_reasonable_trades
            
            # Base penalty: 1000x transaction cost per excess trade (much more severe)
            overtrading_penalty = excess_trades * self.transaction_cost * 1000
            
            # Exponential penalty for severe overtrading (more than 2x reasonable)
            if excess_trades > max_reasonable_trades:
                overtrading_penalty += (excess_trades ** 2) * self.transaction_cost * 10
            
            # Catastrophic penalty for extreme overtrading (>50% trade frequency)
            if n_trades > len(prices) * 0.5:
                overtrading_penalty += abs(total_return) * 10  # Wipe out all gains + penalty
            
            total_return -= overtrading_penalty
            
            # If still trading excessively after penalty, return extreme negative fitness
            # Any trading above max_trade_frequency * 2 is completely unacceptable
            if n_trades > len(prices) * self.max_trade_frequency * 2:
                return -10000.0  # Extreme penalty to eliminate these chromosomes
        
        # Calculate return per trade (reward quality over quantity)
        return_per_trade = total_return / n_trades if n_trades > 0 else 0
        
        # Bonus for good profit factor, scaled by return quality
        profit_bonus = min((profit_factor - 1.0) * return_per_trade * 0.1, abs(total_return) * 0.2)
        
        # Fitness combines total return with return quality and trade efficiency
        fitness = total_return * 0.7 + return_per_trade * n_trades * 0.3 + profit_bonus

        return float(fitness)

    def _simulate_trading(self, signals: np.ndarray, prices: np.ndarray) -> list[float]:
        """
        Simulate trading based on signals and calculate trade returns.

        Args:
            signals: Trading signals (1: buy/long, 0: hold, -1: sell/short)
            prices: Price data for simulation

        Returns:
            List of trade returns (positive for profit, negative for loss)
        """
        trades = []
        position = 0  # 0: no position, 1: long position, -1: short position
        entry_price = 0

        for i in range(len(signals) - self.lookforward_window):
            current_price = prices[i]
            signal = signals[i]

            if signal == 1 and position == 0:
                # Buy signal - enter long position
                position = 1
                entry_price = current_price
            elif signal == -1 and position == 0 and self.enable_short_selling:
                # Sell signal - enter short position
                position = -1
                entry_price = current_price
            elif signal == 0 or i == len(signals) - self.lookforward_window - 1:
                # Hold signal or end of data - close any open position
                if position == 1:
                    # Close long position
                    price_return = (current_price - entry_price) / entry_price
                    trade_return = price_return - 2 * self.transaction_cost  # Buy and sell costs
                    trades.append(trade_return)
                    position = 0
                elif position == -1:
                    # Close short position
                    price_return = (entry_price - current_price) / entry_price  # Inverted for short
                    trade_return = price_return - 2 * self.transaction_cost  # Short and cover costs
                    trades.append(trade_return)
                    position = 0

        return trades

    def _simulate_specialized_trading(self, signals: np.ndarray, prices: np.ndarray, model_type: str = "long") -> list[float]:
        """
        Simulate specialized trading for long or short models.

        Args:
            signals: Binary trading signals (1: trade, 0: hold)
            prices: Price data for simulation
            model_type: "long" for long-only model, "short" for short-only model

        Returns:
            List of trade returns (positive for profit, negative for loss)
        """
        trades = []
        position = 0  # 0: no position, 1: in position
        entry_price = 0

        for i in range(len(signals) - self.lookforward_window):
            current_price = prices[i]
            signal = signals[i]

            if signal == 1 and position == 0:
                # Trade signal - enter position
                position = 1
                entry_price = current_price
            elif (signal == 0 or i == len(signals) - self.lookforward_window - 1) and position == 1:
                # Hold signal or end of data - close position
                position = 0
                
                if model_type == "long":
                    # Long position: profit when price goes up
                    price_return = (current_price - entry_price) / entry_price
                else:  # short
                    # Short position: profit when price goes down
                    price_return = (entry_price - current_price) / entry_price
                
                trade_return = price_return - 2 * self.transaction_cost  # Entry and exit costs
                trades.append(trade_return)

        return trades

    def _create_next_generation(self, population: list[np.ndarray], fitness_scores: np.ndarray) -> list[np.ndarray]:
        """
        Create next generation using selection, crossover, and mutation.

        Args:
            population: Current population (list of chromosomes)
            fitness_scores: Fitness scores for current population

        Returns:
            New population for next generation
        """
        new_population = []

        # Elitism: preserve best solutions
        elite_indices = np.argsort(fitness_scores)[-self.elitism:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Fill rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)

            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            new_population.extend([child1, child2])

        # Trim to exact population size
        return new_population[:self.population_size]

    def _tournament_selection(self, population: list[np.ndarray], fitness_scores: np.ndarray, tournament_size: int = 3) -> np.ndarray:
        """Select parent using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perform single-point crossover."""
        crossover_point = np.random.randint(1, len(parent1))

        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        return child1, child2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Apply binary bit-flip mutation."""
        mutated = chromosome.copy()

        for i in range(len(chromosome)):
            if np.random.random() < self.mutation_rate:
                # Binary mutation: flip bit (0 <-> 1)
                mutated[i] = 1 - mutated[i]

        return mutated

    def _simple_updown_labels(self, prices: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: This method should not be used.
        GA should always return evolved chromosomes, never fallback to simple labeling.

        Args:
            prices: Price array

        Returns:
            Binary labels based on price direction
        """
        raise RuntimeError(
            "GA fallback to simple labeling should never be called. "
            "This indicates a bug in the GA optimization logic."
        )

    def get_target_info(self) -> dict[str, Any]:
        """Return metadata about this generator."""
        if self.dual_models:
            target_names = [f"{self.target_name_prefix}_long_labels", f"{self.target_name_prefix}_short_labels"]
            description = f"Dual GA models: separate long and short optimized strategies with {self.population_size} population, {self.max_generations} generations"
        else:
            target_names = [f"{self.target_name_prefix}_long_labels"]
            description = f"Single GA model with {self.population_size} population, {self.max_generations} generations"
            
        return {
            "target_names": target_names,
            "target_type": "classification",
            "description": description,
            "parameters": {
                "population_size": self.population_size,
                "max_generations": self.max_generations,
                "lookforward_window": self.lookforward_window,
                "transaction_cost": self.transaction_cost,
                "min_trades": self.min_trades,
                "min_win_rate": self.min_win_rate,
                "max_win_rate": self.max_win_rate,
                "min_profit_factor": self.min_profit_factor,
                "elitism": self.elitism,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "dual_models": self.dual_models
            }
        }

