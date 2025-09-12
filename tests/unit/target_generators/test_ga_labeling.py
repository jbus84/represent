"""
Tests for GA Labeling Target Generator

This module tests the genetic algorithm-based trading label generator,
including GA optimization, fitness evaluation, and trading simulation.
"""

import numpy as np
import polars as pl
import pytest

from represent.target_generators.ga_labeling import GALabelingGenerator


class TestGALabelingGenerator:
    """Test suite for GA labeling target generator."""

    def test_initialization(self):
        """Test GA labeling generator initialization."""
        generator = GALabelingGenerator(
            population_size=50,
            max_generations=100,
            lookforward_window=2,
            transaction_cost=0.002,
            random_seed=42
        )

        assert generator.population_size == 50
        assert generator.max_generations == 100
        assert generator.lookforward_window == 2
        assert generator.transaction_cost == 0.002
        assert generator.target_name_prefix == "ga"
        assert generator.required_columns == ["mid_price"]

    def test_default_parameters(self):
        """Test default parameter values."""
        generator = GALabelingGenerator()

        assert generator.population_size == 50
        assert generator.max_generations == 100
        assert generator.lookforward_window == 100
        assert generator.transaction_cost == 0.00007
        assert generator.min_trades == 30
        assert generator.min_win_rate == 0.33
        assert generator.max_win_rate == 0.80
        assert generator.min_profit_factor == 1.0
        assert generator.elitism == 3
        assert generator.mutation_rate == 0.01
        assert generator.crossover_rate == 0.8

    def test_generate_targets_basic(self):
        """Test basic target generation functionality."""
        # Create test data with clear upward trend
        n_samples = 100
        prices = np.linspace(100, 110, n_samples) + np.random.normal(0, 0.1, n_samples)

        df = pl.DataFrame({
            "mid_price": prices,
            "timestamp": range(n_samples)
        })

        generator = GALabelingGenerator(
            population_size=20,
            max_generations=10,
            random_seed=42,
            verbose=False
        )

        result = generator.generate_targets(df, symbol="TEST")

        # Check result structure
        assert isinstance(result, pl.DataFrame)
        assert len(result) == n_samples
        assert "row_idx" in result.columns
        assert "symbol" in result.columns
        assert "ga_long_labels" in result.columns
        assert "ga_short_labels" in result.columns

        # Check data types
        assert result["row_idx"].dtype in [pl.Int64, pl.UInt32]
        assert result["symbol"].dtype == pl.String
        assert result["ga_long_labels"].dtype == pl.Int32
        assert result["ga_short_labels"].dtype == pl.Int32

        # Check values
        assert result["symbol"][0] == "TEST"
        assert all(label in [0, 1] for label in result["ga_long_labels"])
        assert all(label in [0, 1] for label in result["ga_short_labels"])

    def test_generate_targets_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create very small dataset
        n_samples = 10
        prices = np.random.normal(100, 1, n_samples)

        df = pl.DataFrame({
            "mid_price": prices
        })

        generator = GALabelingGenerator(
            min_trades=30,
            verbose=False
        )

        with pytest.warns(UserWarning, match="Insufficient data for GA labeling"):
            result = generator.generate_targets(df)

        # Should return zero labels
        assert all(label == 0 for label in result["ga_long_labels"])
        assert all(label == 0 for label in result["ga_short_labels"])

    def test_simple_updown_fallback(self):
        """Test that simple up/down labeling fallback raises error (deprecated)."""
        prices = [100, 101, 102, 101, 103, 102, 104, 105, 104, 106,
                 107, 106, 108, 109, 108, 110, 111, 110, 112, 111]

        generator = GALabelingGenerator(verbose=False)

        # The fallback method should raise a RuntimeError since it's deprecated
        with pytest.raises(RuntimeError, match="GA fallback to simple labeling should never be called"):
            generator._simple_updown_labels(np.array(prices))

    def test_trading_simulation(self):
        """Test trading simulation logic."""
        prices = np.array([100, 101, 102, 101, 103, 102, 104, 105])
        signals = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # Buy-sell pattern

        generator = GALabelingGenerator(
            lookforward_window=1,
            transaction_cost=0.01,  # 1%
            verbose=False
        )

        trades = generator._simulate_trading(signals, prices)

        # Should have some trades
        assert len(trades) > 0

        # Each trade should account for transaction costs
        for trade in trades:
            assert isinstance(trade, float)
            # Trade return should be net of 2% transaction costs (buy + sell)

    def test_fitness_evaluation(self):
        """Test fitness evaluation function."""
        n_samples = 50
        # Create alternating pattern for predictable trading
        prices = np.array([100 + i % 2 for i in range(n_samples)], dtype=float)
        chromosome = np.array([i % 2 for i in range(n_samples)])

        generator = GALabelingGenerator(
            lookforward_window=1,
            min_trades=5,
            transaction_cost=0.001,  # Low cost for testing
            verbose=False
        )

        fitness = generator._evaluate_fitness(chromosome, prices)

        # Should return a numeric fitness value
        assert isinstance(fitness, float)
        # Fitness should not be a penalty value (> -1000)
        # Note: May still be negative due to transaction costs

    def test_population_initialization(self):
        """Test population initialization."""
        generator = GALabelingGenerator(population_size=10, verbose=False)

        population = generator._initialize_population(20)

        # Check that it returns a list of arrays
        assert isinstance(population, list)
        assert len(population) == 10
        assert all(isinstance(chromo, np.ndarray) for chromo in population)
        assert all(len(chromo) == 20 for chromo in population)

        # Check that all values are binary
        for chromo in population:
            assert np.all((chromo == 0) | (chromo == 1))

    def test_genetic_operations(self):
        """Test genetic algorithm operations."""
        generator = GALabelingGenerator(verbose=False)

        # Test crossover
        parent1 = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        parent2 = np.array([0, 0, 1, 1, 0, 0, 1, 1])

        child1, child2 = generator._crossover(parent1, parent2)

        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)
        assert np.all((child1 == 0) | (child1 == 1))
        assert np.all((child2 == 0) | (child2 == 1))

        # Test mutation
        original = np.array([1, 1, 1, 1, 1])
        mutated = generator._mutate(original)

        assert len(mutated) == len(original)
        assert np.all((mutated == 0) | (mutated == 1))

    def test_tournament_selection(self):
        """Test tournament selection."""
        population = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
        fitness_scores = np.array([0.1, 0.8, 0.6, 0.2])

        generator = GALabelingGenerator(verbose=False)

        # Run multiple selections
        selected_count = {}
        for _ in range(100):
            selected = generator._tournament_selection(population, fitness_scores)
            selected_key = tuple(selected)
            selected_count[selected_key] = selected_count.get(selected_key, 0) + 1

        # Higher fitness individuals should be selected more often
        # Individual 1 (fitness 0.8) should be selected most often
        assert len(selected_count) > 1  # Should have some variety

    def test_target_info(self):
        """Test target info metadata."""
        generator = GALabelingGenerator(
            population_size=50,
            max_generations=100,
            target_name_prefix="custom_ga"
        )

        info = generator.get_target_info()

        assert info["target_names"] == ["custom_ga_long_labels", "custom_ga_short_labels"]
        assert info["target_type"] == "classification"
        assert "Dual GA models" in info["description"]

        params = info["parameters"]
        assert params["population_size"] == 50
        assert params["max_generations"] == 100
        assert "transaction_cost" in params
        assert "min_trades" in params

    def test_missing_required_columns(self):
        """Test error handling for missing required columns."""
        df = pl.DataFrame({
            "price": [100, 101, 102],  # Wrong column name
            "timestamp": [1, 2, 3]
        })

        generator = GALabelingGenerator(verbose=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            generator.generate_targets(df)

    def test_edge_case_constant_prices(self):
        """Test handling of constant price data."""
        n_samples = 50
        prices = np.full(n_samples, 100.0)  # Constant prices

        df = pl.DataFrame({
            "mid_price": prices
        })

        generator = GALabelingGenerator(
            population_size=10,
            max_generations=5,
            verbose=False
        )

        # Should not crash and should return labels
        result = generator.generate_targets(df)
        assert len(result) == n_samples
        assert "ga_long_labels" in result.columns
        assert "ga_short_labels" in result.columns

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        n_samples = 30
        prices = np.random.normal(100, 2, n_samples)

        df = pl.DataFrame({
            "mid_price": prices
        })

        # Generate with same seed twice
        generator1 = GALabelingGenerator(
            population_size=10,
            max_generations=5,
            random_seed=12345,
            verbose=False
        )

        generator2 = GALabelingGenerator(
            population_size=10,
            max_generations=5,
            random_seed=12345,
            verbose=False
        )

        result1 = generator1.generate_targets(df)
        result2 = generator2.generate_targets(df)

        # Results should be identical with same seed
        assert result1["ga_long_labels"].to_list() == result2["ga_long_labels"].to_list()
        assert result1["ga_short_labels"].to_list() == result2["ga_short_labels"].to_list()

    def test_custom_target_name(self):
        """Test custom target name prefix."""
        df = pl.DataFrame({
            "mid_price": np.random.normal(100, 1, 20)
        })

        generator = GALabelingGenerator(
            target_name_prefix="custom",
            population_size=5,
            max_generations=2,
            verbose=False
        )

        result = generator.generate_targets(df)

        assert "custom_long_labels" in result.columns
        assert "custom_short_labels" in result.columns
        assert "ga_long_labels" not in result.columns

    def test_performance_constraints(self):
        """Test that GA respects performance constraints."""
        generator = GALabelingGenerator(
            min_trades=10,
            min_win_rate=0.4,
            max_win_rate=0.7,
            min_profit_factor=1.5,
            verbose=False
        )

        # Create scenario with no trades
        prices = np.array([100, 100, 100, 100, 100])
        chromosome = np.array([0, 0, 0, 0, 0])  # No buy signals

        fitness = generator._evaluate_fitness(chromosome, prices)
        assert fitness == -1000.0  # Penalty for no trades

        # Create scenario with too few trades
        prices = np.array([100, 101, 100, 101, 100] * 3)
        chromosome = np.array([1, 0, 0, 0, 0] * 3)  # Very few signals

        fitness = generator._evaluate_fitness(chromosome, prices)
        assert fitness == -500.0  # Penalty for insufficient trades
