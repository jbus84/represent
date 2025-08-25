#!/usr/bin/env python3
"""
Comprehensive Distribution Testing for Financial Returns Classification

Tests all major heavy-tailed distributions suitable for financial data:
- Cauchy Distribution
- Pareto Distribution
- Hyperbolic Distribution
- Generalized Hyperbolic Distribution
- Variance Gamma Distribution
- Normal Inverse Gaussian (NIG)
- Skewed t-Distribution
"""

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add represent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class DistributionResults:
    """Results for a distribution approach."""

    name: str
    boundaries: np.ndarray
    class_fractions: np.ndarray
    balance_score: float
    extreme_concentration: float
    extreme_excess: float
    parameters: dict[str, Any]
    fit_quality: str
    success: bool


class ComprehensiveDistributionTester:
    """Test all major distributions for financial returns classification."""

    def __init__(self, nbins: int = 13):
        self.nbins = nbins
        self.expected_fraction = 1.0 / nbins

    def fit_cauchy_distribution(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Cauchy Distribution (Lorentz Distribution)
        - Extremely heavy tails (infinite mean and variance)
        - Sometimes used for stress testing
        - Parameters: location, scale
        """
        try:
            # Fit Cauchy distribution
            loc, scale = stats.cauchy.fit(data)

            # Generate boundaries
            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = stats.cauchy.ppf(quantiles, loc=loc, scale=scale)

            # Handle potential infinite values
            if np.any(~np.isfinite(boundaries)):
                # Use percentile-based fallback
                boundaries = np.quantile(data, quantiles)

            params = {"location": loc, "scale": scale, "method": "cauchy_fit"}
            return boundaries, params

        except Exception as e:
            # Fallback to robust fitting
            loc = np.median(data)
            scale = np.percentile(np.abs(data - loc), 50) * 1.4826  # Robust scale

            quantiles = np.linspace(0.01, 0.99, self.nbins + 1)  # Avoid extremes
            boundaries = stats.cauchy.ppf(quantiles, loc=loc, scale=scale)

            params = {"location": loc, "scale": scale, "method": "robust_fallback", "error": str(e)}
            return boundaries, params

    def fit_pareto_distribution(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Pareto Distribution
        - Power-law tail behavior
        - Good for extreme losses/wealth distributions
        - Parameters: shape (α), scale
        """
        try:
            # Pareto requires positive data, so shift if necessary
            data_shift = data - data.min() + 0.001

            # Fit Pareto distribution
            shape, loc, scale = stats.pareto.fit(data_shift, floc=0)

            # Generate boundaries
            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = stats.pareto.ppf(quantiles, shape, loc=0, scale=scale)

            # Shift back to original scale
            boundaries = boundaries + data.min() - 0.001

            params = {
                "shape": shape,
                "scale": scale,
                "data_shift": data.min(),
                "method": "pareto_fit",
            }
            return boundaries, params

        except Exception as e:
            # Fallback to quantile boundaries
            quantiles = np.linspace(0, 1, self.nbins + 1)
            boundaries = np.quantile(data, quantiles)

            params = {"method": "quantile_fallback", "error": str(e)}
            return boundaries, params

    def fit_hyperbolic_distribution(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Hyperbolic Distribution
        - Good for financial returns with skewness and heavy tails
        - More flexible than normal, less extreme than Cauchy
        - Parameters: α (tail heaviness), β (asymmetry), δ (scale), μ (location)
        """
        try:
            # Estimate hyperbolic parameters using method of moments
            mean_val = np.mean(data)
            var_val = np.var(data)
            skew_val = stats.skew(data)
            kurt_val = stats.kurtosis(data, fisher=True)

            # Hyperbolic parameter estimation (approximate)
            # β controls asymmetry (related to skewness)
            beta = skew_val * 0.5

            # α controls tail heaviness (related to kurtosis)
            # Higher kurtosis → higher α (but not too high for numerical stability)
            alpha = max(1.0, min(10.0, 1.0 + kurt_val * 0.1))

            # δ (scale) and μ (location)
            delta = np.sqrt(var_val) * 0.5
            mu = mean_val

            # Generate boundaries using Student's t as approximation
            # (since exact hyperbolic quantiles are complex)
            df = max(2.1, 4.0 / alpha) if alpha > 0 else 2.1

            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = stats.t.ppf(quantiles, df, loc=mu, scale=delta)

            # Apply asymmetry correction
            if abs(beta) > 0.05:
                for i, q in enumerate(quantiles):
                    if q < 0.5:
                        boundaries[i] *= 1.0 - beta * 0.1
                    elif q > 0.5:
                        boundaries[i] *= 1.0 + beta * 0.1

            params = {
                "alpha": alpha,
                "beta": beta,
                "delta": delta,
                "mu": mu,
                "method": "hyperbolic_approximation",
            }
            return boundaries, params

        except Exception as e:
            quantiles = np.linspace(0, 1, self.nbins + 1)
            boundaries = np.quantile(data, quantiles)
            params = {"method": "quantile_fallback", "error": str(e)}
            return boundaries, params

    def fit_variance_gamma_distribution(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Variance Gamma Distribution - OPTIMIZED FOR TAIL PREDICTION
        - Used in option pricing and financial risk modeling
        - Excellent for capturing skewness and leptokurtosis in financial returns
        - Parameters: θ (drift), σ (volatility), ν (variance rate of time)
        - FOCUS: Better tail modeling for classes 0 and 12 prediction
        """
        try:
            # Enhanced VG parameter estimation for better tail behavior
            mean_val = np.mean(data)
            var_val = np.var(data)
            skew_val = stats.skew(data)
            kurt_val = stats.kurtosis(data, fisher=True)

            # VG parameters using improved method of moments
            # ν (variance rate) - controls tail heaviness (higher ν = heavier tails)
            if kurt_val > 3.0:  # Significant excess kurtosis
                nu = max(0.5, min(5.0, kurt_val / 2.0))  # More responsive to kurtosis
            else:
                nu = 0.8  # Moderate tail heaviness

            # θ (drift parameter) - controls asymmetry
            if abs(skew_val) > 0.1:
                theta = skew_val * np.sqrt(var_val) * 0.8  # Stronger skewness response
            else:
                theta = 0.0

            # σ (volatility parameter) - base scale
            sigma = np.sqrt(var_val / (1 + nu * theta**2 / var_val))
            sigma = max(0.01, sigma)  # Ensure positive

            # Generate boundaries using enhanced VG approach
            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = np.zeros(len(quantiles))

            # VG has no closed-form CDF, but we can use gamma-mixture approximation
            # VG(X) ~ Normal(θ*T, σ²*T) where T ~ Gamma(1/ν, ν)

            for i, q in enumerate(quantiles):
                # Enhanced tail modeling for extreme quantiles
                if q <= 0.1 or q >= 0.9:  # Focus on tails for classes 0 and 12
                    # Use gamma-mixture approach for better tail accuracy

                    # Expected gamma time
                    gamma_mean = 1.0  # E[T] = 1/ν * ν = 1

                    # For tail quantiles, use enhanced gamma sampling
                    if q <= 0.05:  # Far left tail
                        # Use more extreme gamma values for left tail
                        gamma_quantile = max(0.1, stats.gamma.ppf(q * 10, a=1 / nu, scale=nu))
                    elif q >= 0.95:  # Far right tail
                        # Use more extreme gamma values for right tail
                        gamma_quantile = stats.gamma.ppf(0.9 + (q - 0.9) * 10, a=1 / nu, scale=nu)
                    else:
                        # Regular tail region
                        gamma_quantile = stats.gamma.ppf(
                            q * 2 if q < 0.5 else (q - 0.5) * 2 + 0.5, a=1 / nu, scale=nu
                        )

                    # Normal component with gamma-scaled parameters
                    normal_mean = theta * gamma_quantile
                    normal_std = sigma * np.sqrt(gamma_quantile)

                    # Calculate boundary using tail-enhanced normal
                    if q < 0.5:
                        # Left tail enhancement
                        tail_quantile = q / 0.5  # Rescale to [0,1]
                        boundary = stats.norm.ppf(
                            tail_quantile * 0.5, loc=normal_mean, scale=normal_std
                        )
                    else:
                        # Right tail enhancement
                        tail_quantile = (q - 0.5) / 0.5  # Rescale to [0,1]
                        boundary = stats.norm.ppf(
                            0.5 + tail_quantile * 0.5, loc=normal_mean, scale=normal_std
                        )

                else:
                    # Center region - use standard approach
                    gamma_mean = 1.0
                    normal_mean = theta * gamma_mean
                    normal_std = sigma * np.sqrt(gamma_mean)
                    boundary = stats.norm.ppf(q, loc=normal_mean, scale=normal_std)

                # Ensure finite boundary
                if not np.isfinite(boundary):
                    boundary = np.quantile(data, q)

                boundaries[i] = boundary

            # Ensure boundaries are sorted and well-spaced
            boundaries = np.sort(boundaries)

            # Enhance tail separation for better class 0/12 prediction
            data_range = boundaries[-1] - boundaries[0]
            min_spacing = data_range / (self.nbins * 100)

            for i in range(1, len(boundaries)):
                if boundaries[i] - boundaries[i - 1] < min_spacing:
                    boundaries[i] = boundaries[i - 1] + min_spacing

            # Extend tails slightly for better extreme coverage
            tail_extension = (boundaries[-1] - boundaries[0]) * 0.1
            boundaries[0] -= tail_extension
            boundaries[-1] += tail_extension

            params = {
                "theta": theta,
                "sigma": sigma,
                "nu": nu,
                "estimated_mean": mean_val,
                "estimated_var": var_val,
                "method": "variance_gamma_optimized_tails",
            }
            return boundaries, params

        except Exception as e:
            # Fallback with enhanced quantile approach for tails
            try:
                # Use enhanced quantile approach that focuses on tails
                base_quantiles = np.linspace(0, 1, self.nbins + 1)

                # Enhance tail resolution
                enhanced_quantiles = []
                for q in base_quantiles:
                    if q <= 0.1:  # Left tail
                        enhanced_q = q * 0.8  # Compress slightly toward center
                    elif q >= 0.9:  # Right tail
                        enhanced_q = 0.9 + (q - 0.9) * 0.8  # Compress slightly toward center
                    else:
                        enhanced_q = q
                    enhanced_quantiles.append(enhanced_q)

                boundaries = np.quantile(data, enhanced_quantiles)
                params = {"method": "enhanced_quantile_fallback", "error": str(e)}
                return boundaries, params

            except Exception:
                # Final fallback
                quantiles = np.linspace(0, 1, self.nbins + 1)
                boundaries = np.quantile(data, quantiles)
                params = {"method": "simple_quantile_fallback", "error": str(e)}
                return boundaries, params

    def fit_nig_distribution(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Normal Inverse Gaussian (NIG) Distribution
        - Very popular for financial returns
        - Captures skewness and heavy tails
        - Parameters: α (tail heaviness), β (asymmetry), δ (scale), μ (location)
        """
        try:
            # NIG parameter estimation using method of moments
            mean_val = np.mean(data)
            var_val = np.var(data)
            skew_val = stats.skew(data)
            kurt_val = stats.kurtosis(data, fisher=True)

            # NIG parameters (method of moments approximation)
            # β controls asymmetry
            beta = skew_val * np.sqrt(var_val) * 0.3

            # α controls tail heaviness (α > |β|)
            alpha = max(abs(beta) + 0.5, np.sqrt(3 + kurt_val) * np.sqrt(var_val))

            # δ scale parameter
            delta = np.sqrt(var_val) * 0.7

            # μ location parameter
            mu = mean_val - beta * delta**2 / alpha

            # Use generalized hyperbolic approximation (NIG is special case)
            # Approximate with Student's t adjusted for NIG characteristics
            df_equiv = max(2.1, 6.0 / kurt_val if kurt_val > 0.5 else 8.0)

            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = stats.t.ppf(quantiles, df_equiv, loc=mu, scale=delta)

            # Apply NIG-specific asymmetry correction
            if abs(beta) > 0.1:
                # NIG has specific skewness pattern
                asymmetry_correction = beta * delta / alpha * 0.5
                for i, q in enumerate(quantiles):
                    if q != 0.5:
                        boundaries[i] += asymmetry_correction * np.log(q / (1 - q)) * 0.1

            params = {
                "alpha": alpha,
                "beta": beta,
                "delta": delta,
                "mu": mu,
                "df_equiv": df_equiv,
                "method": "nig_approximation",
            }
            return boundaries, params

        except Exception as e:
            quantiles = np.linspace(0, 1, self.nbins + 1)
            boundaries = np.quantile(data, quantiles)
            params = {"method": "quantile_fallback", "error": str(e)}
            return boundaries, params

    def fit_skewed_t_distribution(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Skewed t-Distribution
        - Allows asymmetric fat tails
        - Very flexible for financial returns
        - Parameters: df (degrees of freedom), skewness, location, scale
        """
        try:
            # Estimate skewed t-distribution parameters
            mean_val = np.mean(data)
            var_val = np.var(data)
            skew_val = stats.skew(data)
            kurt_val = stats.kurtosis(data, fisher=True)

            # Degrees of freedom from kurtosis
            if kurt_val > 1.0:
                df = max(2.1, 6.0 / kurt_val + 4.0)
            else:
                df = 10.0

            # Skewness parameter
            skewness_param = skew_val * 0.5

            # Location and scale
            loc = mean_val
            scale = np.sqrt(var_val * (df - 2) / df) if df > 2 else np.sqrt(var_val)

            # Generate boundaries using skewed t approach
            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)

            # Base t-distribution boundaries
            boundaries = stats.t.ppf(quantiles, df, loc=loc, scale=scale)

            # Apply skewness transformation
            if abs(skewness_param) > 0.05:
                # Skewed t-distribution transformation
                for i, q in enumerate(quantiles):
                    # Transform quantile based on skewness
                    if q < 0.5:
                        # Left tail
                        skewed_q = 0.5 * (2 * q) ** (1 + skewness_param)
                    else:
                        # Right tail
                        skewed_q = 1 - 0.5 * (2 * (1 - q)) ** (1 - skewness_param)

                    # Recalculate boundary with skewed quantile
                    if 0.001 <= skewed_q <= 0.999:
                        boundaries[i] = stats.t.ppf(skewed_q, df, loc=loc, scale=scale)

            params = {
                "df": df,
                "skewness": skewness_param,
                "location": loc,
                "scale": scale,
                "method": "skewed_t_fit",
            }
            return boundaries, params

        except Exception as e:
            # Fallback to regular t-distribution
            try:
                df, loc, scale = stats.t.fit(data)
                df = max(2.1, min(30, df))

                quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
                boundaries = stats.t.ppf(quantiles, df, loc=loc, scale=scale)

                params = {"df": df, "location": loc, "scale": scale, "method": "t_fallback"}
                return boundaries, params

            except Exception:
                quantiles = np.linspace(0, 1, self.nbins + 1)
                boundaries = np.quantile(data, quantiles)
                params = {"method": "quantile_fallback", "error": str(e)}
                return boundaries, params

    def fit_merton_jump_diffusion(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Merton's Jump Diffusion Model (Poisson Jump Models)
        - Models asset returns with rare jumps
        - Combines continuous diffusion with discrete jumps
        - Parameters: λ (jump intensity), μ_j (jump mean), σ_j (jump std), σ (diffusion vol)
        """
        try:
            # Estimate Merton Jump Diffusion parameters
            mean_val = np.mean(data)
            var_val = np.var(data)
            skew_val = stats.skew(data)
            kurt_val = stats.kurtosis(data, fisher=True)

            # Jump detection using kurtosis - high kurtosis suggests jumps
            excess_kurtosis = kurt_val
            if excess_kurtosis > 3.0:  # Evidence of jumps
                # Jump intensity (λ) - more jumps for higher kurtosis
                jump_intensity = min(0.5, excess_kurtosis / 20.0)

                # Jump parameters
                jump_mean = skew_val * np.sqrt(var_val) * 0.2  # Jump direction from skewness
                jump_std = np.sqrt(var_val) * 0.8  # Jump magnitude

                # Diffusion volatility (after removing jump contribution)
                diffusion_vol = np.sqrt(max(0.1 * var_val, var_val - jump_intensity * jump_std**2))
            else:
                # Low kurtosis - minimal jumps
                jump_intensity = 0.05
                jump_mean = 0.0
                jump_std = np.sqrt(var_val) * 0.3
                diffusion_vol = np.sqrt(var_val)

            # Generate boundaries using compound distribution
            # Merton model: X = Normal(diffusion) + Poisson(λ) * Normal(jump_mean, jump_std)
            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = np.zeros(len(quantiles))

            for i, q in enumerate(quantiles):
                # Approximate compound distribution quantiles
                # Using weighted mixture of diffusion and jump components

                # Base diffusion component
                diffusion_quantile = stats.norm.ppf(q, loc=mean_val, scale=diffusion_vol)

                # Jump contribution (approximated)
                jump_contribution = 0.0
                if jump_intensity > 0.01:  # Only if significant jumps
                    # Expected jump contribution
                    expected_jumps = jump_intensity
                    jump_contribution = expected_jumps * jump_mean

                    # Tail enhancement for extreme quantiles
                    if q < 0.1 or q > 0.9:
                        # Enhance tails with jump volatility
                        tail_enhancement = jump_std * stats.norm.ppf(q) * 0.3
                        jump_contribution += tail_enhancement

                boundaries[i] = diffusion_quantile + jump_contribution

            # Ensure boundaries are sorted
            boundaries = np.sort(boundaries)

            params = {
                "jump_intensity": jump_intensity,
                "jump_mean": jump_mean,
                "jump_std": jump_std,
                "diffusion_vol": diffusion_vol,
                "method": "merton_jump_diffusion",
            }
            return boundaries, params

        except Exception as e:
            # Fallback to enhanced normal distribution
            mean_val = np.mean(data)
            std_val = np.std(data)

            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = stats.norm.ppf(quantiles, loc=mean_val, scale=std_val)

            params = {"method": "normal_fallback", "error": str(e)}
            return boundaries, params

    def fit_double_exponential_jump_diffusion(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Double Exponential Jump Diffusion (Kou Model)
        - Better fit for sudden price moves with asymmetric jumps
        - Uses double exponential distribution for jumps
        - Parameters: λ (intensity), p (up-jump prob), η₁ (up decay), η₂ (down decay)
        """
        try:
            mean_val = np.mean(data)
            var_val = np.var(data)
            skew_val = stats.skew(data)
            kurt_val = stats.kurtosis(data, fisher=True)

            # Jump detection and parameter estimation
            excess_kurtosis = kurt_val

            if excess_kurtosis > 3.0:  # Evidence of jumps
                # Jump intensity
                jump_intensity = min(0.4, excess_kurtosis / 25.0)

                # Asymmetric jump parameters based on skewness
                if skew_val > 0.1:
                    # Positive skew - more upward jumps
                    up_jump_prob = 0.6
                    up_decay_rate = 10.0  # η₁ - controls upward jump tail
                    down_decay_rate = 15.0  # η₂ - controls downward jump tail
                elif skew_val < -0.1:
                    # Negative skew - more downward jumps
                    up_jump_prob = 0.4
                    up_decay_rate = 15.0
                    down_decay_rate = 10.0
                else:
                    # Symmetric jumps
                    up_jump_prob = 0.5
                    up_decay_rate = 12.0
                    down_decay_rate = 12.0

                # Diffusion component
                diffusion_vol = np.sqrt(max(0.1 * var_val, var_val * 0.6))
            else:
                # Minimal jumps
                jump_intensity = 0.02
                up_jump_prob = 0.5
                up_decay_rate = 10.0
                down_decay_rate = 10.0
                diffusion_vol = np.sqrt(var_val)

            # Generate boundaries for double exponential jump model
            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = np.zeros(len(quantiles))

            for i, q in enumerate(quantiles):
                # Base diffusion component
                diffusion_quantile = stats.norm.ppf(q, loc=mean_val, scale=diffusion_vol)

                # Double exponential jump adjustment
                jump_adjustment = 0.0
                if jump_intensity > 0.01:
                    if q < 0.5:
                        # Left tail - negative jumps
                        tail_distance = 0.5 - q
                        jump_size = -1.0 / down_decay_rate * np.log(1 - 2 * tail_distance)
                        jump_adjustment = jump_intensity * (1 - up_jump_prob) * jump_size
                    else:
                        # Right tail - positive jumps
                        tail_distance = q - 0.5
                        jump_size = 1.0 / up_decay_rate * np.log(1 + 2 * tail_distance)
                        jump_adjustment = jump_intensity * up_jump_prob * jump_size

                boundaries[i] = diffusion_quantile + jump_adjustment

            # Ensure boundaries are sorted
            boundaries = np.sort(boundaries)

            params = {
                "jump_intensity": jump_intensity,
                "up_jump_prob": up_jump_prob,
                "up_decay_rate": up_decay_rate,
                "down_decay_rate": down_decay_rate,
                "diffusion_vol": diffusion_vol,
                "method": "double_exp_jump_diffusion",
            }
            return boundaries, params

        except Exception as e:
            # Fallback
            mean_val = np.mean(data)
            std_val = np.std(data)

            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = stats.norm.ppf(quantiles, loc=mean_val, scale=std_val)

            params = {"method": "normal_fallback", "error": str(e)}
            return boundaries, params

    def fit_mixture_of_normals(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Mixture of Normals
        - Captures volatility clustering in returns
        - Models different market regimes with separate normal distributions
        - Parameters: weights, means, variances for each component
        """
        try:
            # Use Gaussian Mixture Model to fit data
            # Start with 2-3 components to capture low/high volatility regimes
            best_aic = np.inf
            best_model = None
            best_n_components = 2

            # Try different numbers of components
            for n_components in [2, 3]:
                try:
                    model = GaussianMixture(
                        n_components=n_components,
                        covariance_type="full",
                        random_state=42,
                        max_iter=200,
                    )
                    model.fit(data.reshape(-1, 1))

                    aic = model.aic(data.reshape(-1, 1))
                    if aic < best_aic:
                        best_aic = aic
                        best_model = model
                        best_n_components = n_components

                except Exception:
                    continue

            if best_model is None:
                raise ValueError("Could not fit mixture model")

            # Extract parameters
            weights = best_model.weights_
            means = best_model.means_.flatten()
            covariances = best_model.covariances_.flatten()
            stds = np.sqrt(covariances)

            # Generate boundaries using mixture distribution
            def mixture_cdf(x):
                """Cumulative distribution function for mixture"""
                cdf_val = 0.0
                for i in range(best_n_components):
                    cdf_val += weights[i] * stats.norm.cdf(x, loc=means[i], scale=stds[i])
                return cdf_val

            def mixture_ppf(q):
                """Percent point function (inverse CDF) for mixture"""
                # Use numerical method to find quantile
                from scipy.optimize import brentq

                # Determine search range
                data_min = np.min(data)
                data_max = np.max(data)
                data_range = data_max - data_min
                search_min = data_min - data_range
                search_max = data_max + data_range

                try:
                    return brentq(lambda x: mixture_cdf(x) - q, search_min, search_max)
                except Exception:
                    # Fallback to simple interpolation
                    return np.quantile(data, q)

            # Generate boundaries
            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = np.array([mixture_ppf(q) for q in quantiles])

            # Ensure boundaries are finite and sorted
            boundaries = np.where(np.isfinite(boundaries), boundaries, np.quantile(data, quantiles))
            boundaries = np.sort(boundaries)

            params = {
                "n_components": best_n_components,
                "weights": weights.tolist(),
                "means": means.tolist(),
                "stds": stds.tolist(),
                "aic": best_aic,
                "method": "mixture_of_normals",
            }
            return boundaries, params

        except Exception as e:
            # Fallback to quantiles
            quantiles = np.linspace(0, 1, self.nbins + 1)
            boundaries = np.quantile(data, quantiles)
            params = {"method": "quantile_fallback", "error": str(e)}
            return boundaries, params

    def fit_markov_switching_distribution(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Markov-switching distributions
        - Regime-dependent distributions (e.g., bull vs. bear markets)
        - Uses time series structure to identify regime changes
        - Parameters: regime probabilities, regime-specific distributions
        """
        try:
            # Simple regime detection based on rolling volatility
            # More sophisticated approaches would use Hidden Markov Models

            # Calculate rolling volatility to identify regimes
            window_size = min(1000, len(data) // 10)
            rolling_vol = np.array(
                [np.std(data[max(0, i - window_size) : i + 1]) for i in range(len(data))]
            )

            # Identify high/low volatility regimes
            vol_threshold = np.median(rolling_vol)

            # High volatility regime (bear/crisis)
            high_vol_mask = rolling_vol >= vol_threshold
            high_vol_data = data[high_vol_mask]

            # Low volatility regime (bull/calm)
            low_vol_mask = rolling_vol < vol_threshold
            low_vol_data = data[low_vol_mask]

            # Regime probabilities
            high_vol_prob = np.mean(high_vol_mask)
            low_vol_prob = 1 - high_vol_prob

            if len(high_vol_data) < 100 or len(low_vol_data) < 100:
                raise ValueError("Insufficient data for regime identification")

            # Fit distributions for each regime
            # High volatility regime - use t-distribution
            high_df, high_loc, high_scale = stats.t.fit(high_vol_data)
            high_df = max(2.1, min(30, high_df))

            # Low volatility regime - use normal or light-tailed t
            low_df, low_loc, low_scale = stats.t.fit(low_vol_data)
            low_df = max(5.0, min(50, low_df))  # Less heavy tails for calm periods

            # Generate boundaries using regime-weighted mixture
            def regime_mixture_cdf(x):
                """CDF for regime-switching model"""
                high_cdf = stats.t.cdf(x, high_df, loc=high_loc, scale=high_scale)
                low_cdf = stats.t.cdf(x, low_df, loc=low_loc, scale=low_scale)
                return high_vol_prob * high_cdf + low_vol_prob * low_cdf

            def regime_mixture_ppf(q):
                """PPF for regime-switching model"""
                from scipy.optimize import brentq

                data_range = np.max(data) - np.min(data)
                search_min = np.min(data) - 2 * data_range
                search_max = np.max(data) + 2 * data_range

                try:
                    return brentq(lambda x: regime_mixture_cdf(x) - q, search_min, search_max)
                except Exception:
                    # Weighted combination fallback
                    high_quantile = stats.t.ppf(q, high_df, loc=high_loc, scale=high_scale)
                    low_quantile = stats.t.ppf(q, low_df, loc=low_loc, scale=low_scale)
                    return high_vol_prob * high_quantile + low_vol_prob * low_quantile

            # Generate boundaries
            quantiles = np.linspace(0.001, 0.999, self.nbins + 1)
            boundaries = np.array([regime_mixture_ppf(q) for q in quantiles])

            # Ensure finite and sorted
            boundaries = np.where(np.isfinite(boundaries), boundaries, np.quantile(data, quantiles))
            boundaries = np.sort(boundaries)

            params = {
                "high_vol_prob": high_vol_prob,
                "low_vol_prob": low_vol_prob,
                "high_regime": {"df": high_df, "loc": high_loc, "scale": high_scale},
                "low_regime": {"df": low_df, "loc": low_loc, "scale": low_scale},
                "vol_threshold": vol_threshold,
                "method": "markov_switching",
            }
            return boundaries, params

        except Exception as e:
            # Fallback
            quantiles = np.linspace(0, 1, self.nbins + 1)
            boundaries = np.quantile(data, quantiles)
            params = {"method": "quantile_fallback", "error": str(e)}
            return boundaries, params

    def test_distribution(
        self, name: str, fit_func, sample_data: np.ndarray, validation_data: np.ndarray
    ) -> DistributionResults:
        """Test a single distribution approach."""

        print(f"\n🔬 Testing {name}...")

        try:
            # Fit distribution and get boundaries
            boundaries, params = fit_func(sample_data)

            # Ensure finite boundaries
            if not np.all(np.isfinite(boundaries)):
                raise ValueError("Non-finite boundaries generated")

            # Test on validation data
            labels = np.digitize(validation_data, boundaries[1:-1])
            labels = np.clip(labels, 0, self.nbins - 1)

            class_counts = np.bincount(labels, minlength=self.nbins)
            class_fractions = class_counts / len(validation_data)

            # Calculate metrics
            deviations = np.abs(class_fractions - self.expected_fraction)
            max_deviation = np.max(deviations)
            balance_score = 1.0 - (max_deviation / self.expected_fraction)

            extreme_concentration = class_fractions[0] + class_fractions[self.nbins - 1]
            expected_extreme = 2 * self.expected_fraction
            extreme_excess = extreme_concentration - expected_extreme

            # Determine fit quality
            if balance_score > 0.0:
                fit_quality = "good"
                success = True
            elif balance_score > -1.0:
                fit_quality = "acceptable"
                success = True
            else:
                fit_quality = "poor"
                success = False

            print(f"   Balance Score: {balance_score:.3f}")
            print(f"   Extreme Classes (0+{self.nbins - 1}): {extreme_concentration * 100:.1f}%")
            print(f"   Extreme Excess: {extreme_excess * 100:+.1f} pp")
            print(f"   Fit Quality: {fit_quality}")

            return DistributionResults(
                name=name,
                boundaries=boundaries,
                class_fractions=class_fractions,
                balance_score=balance_score,
                extreme_concentration=extreme_concentration,
                extreme_excess=extreme_excess,
                parameters=params,
                fit_quality=fit_quality,
                success=success,
            )

        except Exception as e:
            print(f"   ❌ Failed: {e}")

            # Return failed result with quantile fallback
            quantiles = np.linspace(0, 1, self.nbins + 1)
            boundaries = np.quantile(sample_data, quantiles)

            return DistributionResults(
                name=f"{name} (FAILED)",
                boundaries=boundaries,
                class_fractions=np.full(self.nbins, self.expected_fraction),
                balance_score=-10.0,
                extreme_concentration=2 * self.expected_fraction,
                extreme_excess=0.0,
                parameters={"error": str(e)},
                fit_quality="failed",
                success=False,
            )

    def run_comprehensive_test(
        self, sample_data: np.ndarray, validation_data: np.ndarray
    ) -> dict[str, DistributionResults]:
        """Run comprehensive test of all distributions."""

        print("🎯 COMPREHENSIVE DISTRIBUTION TESTING")
        print("=" * 60)
        print("Testing all major heavy-tailed distributions for financial returns")
        print(f"Sample: {len(sample_data):,}, Validation: {len(validation_data):,}")

        # Define all distributions to test
        distributions = [
            ("Cauchy", self.fit_cauchy_distribution),
            ("Pareto", self.fit_pareto_distribution),
            ("Hyperbolic", self.fit_hyperbolic_distribution),
            ("Variance Gamma", self.fit_variance_gamma_distribution),
            ("NIG (Normal Inverse Gaussian)", self.fit_nig_distribution),
            ("Skewed t-Distribution", self.fit_skewed_t_distribution),
            ("Merton Jump Diffusion", self.fit_merton_jump_diffusion),
            ("Double Exp Jump Diffusion", self.fit_double_exponential_jump_diffusion),
            ("Mixture of Normals", self.fit_mixture_of_normals),
            ("Markov Switching", self.fit_markov_switching_distribution),
        ]

        # Add baseline for comparison
        def fit_quantile_baseline(data):
            quantiles = np.linspace(0, 1, self.nbins + 1)
            boundaries = np.quantile(data, quantiles)
            params = {"method": "quantile"}
            return boundaries, params

        distributions.append(("Quantile (Baseline)", fit_quantile_baseline))

        # Test all distributions
        results = {}
        for name, fit_func in distributions:
            result = self.test_distribution(name, fit_func, sample_data, validation_data)
            results[name] = result

        return results


def main():
    """Run comprehensive distribution testing."""

    # Load data
    print("📊 Loading AUDUSD price movement data...")
    df = pl.read_parquet(
        "/Users/danielfisher/data/databento/AUDUSD_classified_datasets/AUDUSD_M6AM4_dataset.parquet"
    )

    sample_movements = df["price_movement"].to_numpy()[:100000]
    validation_movements = df["price_movement"].to_numpy()[100000:]

    print(f"   Sample size: {len(sample_movements):,}")
    print(f"   Validation size: {len(validation_movements):,}")
    print("   Data characteristics:")
    print(f"      Mean: {sample_movements.mean():.6f}")
    print(f"      Std: {sample_movements.std():.6f}")
    print(f"      Skewness: {stats.skew(sample_movements):.3f}")
    print(f"      Kurtosis: {stats.kurtosis(sample_movements, fisher=True):.3f}")

    # Run comprehensive test
    tester = ComprehensiveDistributionTester(nbins=13)
    results = tester.run_comprehensive_test(sample_movements, validation_movements)

    # Summary results
    print("\n" + "=" * 60)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 60)

    print("\nDistribution Performance Ranking:")
    print("Rank | Distribution | Balance Score | Extreme (0+12) | Fit Quality")
    print("-----|--------------|---------------|-----------------|------------")

    # Sort by balance score (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1].balance_score, reverse=True)

    for i, (name, result) in enumerate(sorted_results, 1):
        print(
            f"{i:4d} | {name[:12]:12} | {result.balance_score:11.3f} | {result.extreme_concentration * 100:13.1f}% | {result.fit_quality:11}"
        )

    # Find best performing distributions
    successful_results = [
        (name, result)
        for name, result in results.items()
        if result.success and result.balance_score > 0
    ]

    if successful_results:
        print("\n✅ SUCCESSFUL APPROACHES:")

        # Sort successful by balance score
        successful_results.sort(key=lambda x: x[1].balance_score, reverse=True)
        best_name, best_result = successful_results[0]

        print(f"\n🏆 WINNER: {best_name}")
        print(f"   Balance Score: {best_result.balance_score:.3f}")
        print(f"   Extreme Concentration: {best_result.extreme_concentration * 100:.1f}%")
        print(f"   Parameters: {best_result.parameters}")

        # Compare top 3
        print("\n📊 TOP 3 COMPARISON:")
        print("Distribution | Balance | Extreme | Key Insight")
        print("-------------|---------|---------|-------------")

        for name, result in successful_results[:3]:
            key_insight = (
                "Heavy tails"
                if "cauchy" in name.lower()
                else "Power law"
                if "pareto" in name.lower()
                else "Skew modeling"
                if "skewed" in name.lower()
                else "Flexible tails"
            )

            print(
                f"{name[:11]:11} | {result.balance_score:7.3f} | {result.extreme_concentration * 100:6.1f}% | {key_insight}"
            )

        return successful_results[0]
    else:
        print("\n❌ NO SUCCESSFUL APPROACHES")
        print("All distributions failed to improve upon baseline quantile approach")
        return None


if __name__ == "__main__":
    try:
        best_result = main()
        if best_result:
            print(f"\n🎉 Best distribution found: {best_result[0]}")
        else:
            print("\n⚠️  No improvements found over baseline quantile approach.")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback

        traceback.print_exc()
