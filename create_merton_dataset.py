#!/usr/bin/env python3
"""
Create Merton Jump Diffusion Classified Dataset

This script creates a new classified dataset using the optimal Merton Jump Diffusion
approach identified through comprehensive distribution analysis, replacing the
traditional quantile-based classification.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

# Add represent to path
sys.path.insert(0, str(Path(__file__).parent))

# Add distributions to path for Merton implementation
sys.path.insert(0, str(Path(__file__).parent / "distributions"))

from represent import DatasetBuildConfig, build_datasets_from_dbn_files, create_represent_config


class MertonJumpDiffusionClassifier:
    """
    Merton Jump Diffusion classifier for financial returns.

    Based on comprehensive distribution analysis showing this approach provides
    the best tail prediction accuracy (tail score: 4.6 vs 14.4 baseline).
    """

    def __init__(self, nbins: int = 13):
        self.nbins = nbins

    def fit_merton_jump_diffusion(self, data: np.ndarray):
        """
        Fit Merton Jump Diffusion model to financial returns data.

        Model: dS/S = μdt + σdW + (e^J - 1)dN
        Where:
        - μ: drift
        - σ: volatility (diffusion)
        - J: jump size (normally distributed)
        - N: Poisson process (jump frequency)
        """
        # Clean data
        finite_data = data[np.isfinite(data)]
        if len(finite_data) < 100:
            raise ValueError("Insufficient finite data for fitting")

        # Calculate moments for parameter estimation
        mean_return = np.mean(finite_data)
        var_return = np.var(finite_data)
        skew_return = stats.skew(finite_data)
        kurt_return = stats.kurtosis(finite_data)

        # Detect jump characteristics using excess kurtosis
        excess_kurtosis = kurt_return

        if excess_kurtosis > 3.0:  # Evidence of jumps
            # Jump intensity estimation (jumps per period)
            jump_intensity = min(0.5, excess_kurtosis / 20.0)

            # Jump size variance (estimated from excess kurtosis)
            jump_var = max(0.01, (excess_kurtosis - 3.0) / 100.0)

            # Diffusion volatility (adjusted for jumps)
            diffusion_vol = float(
                np.sqrt(max(0.001, float(var_return) - jump_intensity * jump_var))
            )

        else:
            # Low kurtosis - minimal jumps, mostly diffusion
            jump_intensity = 0.05
            jump_var = 0.01
            diffusion_vol = np.sqrt(var_return)

        # Jump mean (typically negative for financial data - crashes)
        jump_mean = skew_return * 0.1 if abs(skew_return) > 0.5 else -0.02

        # Drift estimation
        drift = mean_return - jump_intensity * (np.exp(jump_mean + 0.5 * jump_var) - 1)

        return {
            "drift": drift,
            "diffusion_vol": diffusion_vol,
            "jump_intensity": jump_intensity,
            "jump_mean": jump_mean,
            "jump_var": jump_var,
            "method": "merton_jump_diffusion",
        }

    def create_merton_boundaries(self, sample_data: np.ndarray):
        """Create classification boundaries using Merton Jump Diffusion model."""

        # Fit model parameters
        params = self.fit_merton_jump_diffusion(sample_data)

        print("   📊 Merton Parameters:")
        print(f"      Drift: {params['drift']:.6f}")
        print(f"      Diffusion Vol: {params['diffusion_vol']:.6f}")
        print(f"      Jump Intensity: {params['jump_intensity']:.3f}")
        print(f"      Jump Mean: {params['jump_mean']:.6f}")
        print(f"      Jump Variance: {params['jump_var']:.6f}")

        # Generate quantiles for boundaries
        quantiles = np.linspace(0, 1, self.nbins + 1)
        boundaries = []

        for q in quantiles:
            if q == 0:
                # Extreme negative boundary
                boundary = sample_data.min() - abs(sample_data.min()) * 0.1
            elif q == 1:
                # Extreme positive boundary
                boundary = sample_data.max() + abs(sample_data.max()) * 0.1
            else:
                # Model-based quantile estimation
                # For Merton model, we approximate the distribution as a mixture
                # of normal (diffusion) and jump components

                # Base diffusion component
                diffusion_component = stats.norm.ppf(
                    q, loc=params["drift"], scale=params["diffusion_vol"]
                )

                # Jump component contribution
                if params["jump_intensity"] > 0.1 and (q <= 0.1 or q >= 0.9):
                    # Enhanced tail modeling for extreme quantiles
                    if q <= 0.1:
                        # Negative tail - emphasize crash risk
                        jump_contribution = params["jump_mean"] * params["jump_intensity"] * 2.0
                    else:
                        # Positive tail - moderate jump upside
                        jump_contribution = params["jump_mean"] * params["jump_intensity"] * 0.5

                    boundary = diffusion_component + jump_contribution
                else:
                    # Center region - primarily diffusion
                    boundary = diffusion_component

            boundaries.append(boundary)

        boundaries = np.array(sorted(boundaries))

        # Ensure reasonable spacing
        min_spacing = (boundaries[-1] - boundaries[0]) / (len(boundaries) * 100)
        for i in range(1, len(boundaries)):
            if boundaries[i] - boundaries[i - 1] < min_spacing:
                boundaries[i] = boundaries[i - 1] + min_spacing

        return boundaries, params


def create_merton_classified_dataset():
    """Create a new classified dataset using Merton Jump Diffusion approach."""

    print("🎯 CREATING MERTON JUMP DIFFUSION CLASSIFIED DATASET")
    print("=" * 60)
    print("🌟 Using optimal approach from distribution analysis")
    print("📈 Expected: 68% better tail prediction vs baseline")
    print("🎲 Tail Score Target: ~4.6 (vs 14.4 baseline)")
    print()

    # Create configuration
    print("📝 Creating represent configuration...")
    config = create_represent_config(
        currency="AUDUSD",
        features=["volume"],
        lookback_rows=5000,
        lookforward_input=5000,
        lookforward_offset=500,
        jump_size=100,
        nbins=13,
    )
    dataset_cfg, threshold_cfg, processor_cfg = config
    print(f"   Currency: {dataset_cfg.currency}")
    print(f"   Features: {processor_cfg.features}")
    print(f"   Bins: {threshold_cfg.nbins}")
    print()

    # Dataset configuration
    dataset_config = DatasetBuildConfig(
        currency="AUDUSD", min_symbol_samples=60500, force_uniform=True, keep_intermediate=False
    )

    # Select comprehensive DBN file set (first 20 files for good coverage)
    dbn_directory = Path("/Users/danielfisher/data/databento/AUDUSD-micro")
    dbn_files = sorted(dbn_directory.glob("*.dbn.zst"))[:20]

    print(f"📂 Selected {len(dbn_files)} DBN files for comprehensive dataset:")
    for i, file in enumerate(dbn_files[:5]):
        print(f"   {i + 1}. {file.name}")
    if len(dbn_files) > 5:
        print(f"   ... and {len(dbn_files) - 5} more files")
    print()

    # Output to new directory
    output_dir = "/Users/danielfisher/data/databento/AUDUSD_merton_datasets/"
    Path(output_dir).mkdir(exist_ok=True)

    print("🔧 PHASE 1: Standard Symbol-Split-Merge Processing")
    print("-" * 50)

    # First, create standard datasets using existing workflow
    build_datasets_from_dbn_files(
        config=config,
        dbn_files=[str(f) for f in dbn_files],
        output_dir=output_dir,
        dataset_config=dataset_config,
        verbose=True,
    )

    print("\n🎯 PHASE 2: Apply Merton Jump Diffusion Classification")
    print("-" * 50)

    # Now re-classify using Merton approach
    merton_classifier = MertonJumpDiffusionClassifier(nbins=13)

    # Process each created dataset
    output_path = Path(output_dir)
    datasets = list(output_path.glob("AUDUSD_*.parquet"))

    merton_results = []

    for dataset_file in datasets:
        print(f"\n📊 Applying Merton classification to: {dataset_file.name}")

        # Load dataset
        df = pl.read_parquet(dataset_file)
        print(f"   Loaded: {len(df):,} samples")

        # Extract price movements for reclassification
        price_movements = df["price_movement"].to_numpy()
        valid_movements = price_movements[np.isfinite(price_movements)]

        if len(valid_movements) < 10000:
            print(
                f"   ⚠️  Skipping {dataset_file.name} - insufficient samples ({len(valid_movements)})"
            )
            continue

        print(f"   Valid movements: {len(valid_movements):,}")

        # Create Merton-based boundaries using first half (no data leakage)
        training_size = len(valid_movements) // 2
        training_movements = valid_movements[:training_size]

        print(f"   Training sample: {len(training_movements):,} movements")

        # Generate Merton boundaries
        merton_boundaries, merton_params = merton_classifier.create_merton_boundaries(
            training_movements
        )

        # Apply to all data
        new_labels = np.digitize(valid_movements, merton_boundaries[1:-1])
        new_labels = np.clip(new_labels, 0, 12)

        # Update the dataframe with new classification
        df_updated = df.with_columns(
            [pl.Series("classification_label_merton", new_labels).cast(pl.Int8)]
        )

        # Analyze new distribution
        class_counts = np.bincount(new_labels, minlength=13)
        class_fractions = class_counts / len(valid_movements)

        # Calculate metrics
        expected_fraction = 1.0 / 13
        deviations = np.abs(class_fractions - expected_fraction)
        max_deviation = np.max(deviations)
        balance_score = 1.0 - (max_deviation / expected_fraction)

        extreme_concentration = class_fractions[0] + class_fractions[12]
        extreme_excess = extreme_concentration - (2 * expected_fraction)

        print("   🎯 Merton Classification Results:")
        print(f"      Balance Score: {balance_score:.3f}")
        print(f"      Extreme Classes (0+12): {extreme_concentration * 100:.1f}%")
        print(f"      Extreme Excess: {extreme_excess * 100:+.1f} pp")

        # Save updated dataset
        merton_file = output_path / f"{dataset_file.stem}_merton.parquet"
        df_updated.write_parquet(merton_file)
        print(f"   💾 Saved: {merton_file.name}")

        # Store results for comparison
        merton_results.append(
            {
                "symbol": dataset_file.stem.split("_")[-2],  # Extract symbol
                "original_file": dataset_file.name,
                "merton_file": merton_file.name,
                "samples": len(valid_movements),
                "balance_score": balance_score,
                "extreme_concentration": extreme_concentration,
                "extreme_excess": extreme_excess,
                "class_distribution": class_fractions.tolist(),
                "parameters": merton_params,
            }
        )

    print("\n✅ MERTON CLASSIFICATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Processed {len(merton_results)} symbol datasets")
    print(f"📁 Output directory: {output_dir}")
    print("🌟 Merton-classified datasets ready for ML training!")

    return merton_results, output_dir


if __name__ == "__main__":
    try:
        results, output_dir = create_merton_classified_dataset()

        if results:
            print("\n📈 SUMMARY:")
            avg_balance = np.mean([r["balance_score"] for r in results])
            avg_extreme = np.mean([r["extreme_concentration"] for r in results])

            print(f"   Average Balance Score: {avg_balance:.3f}")
            print(f"   Average Extreme Concentration: {avg_extreme * 100:.1f}%")
            print("   Expected Improvement: Superior tail prediction for trading")

        print(f"\n🎉 SUCCESS: Merton-based datasets created in {output_dir}")

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
