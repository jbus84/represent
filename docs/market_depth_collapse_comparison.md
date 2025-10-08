# Market Depth Collapse Comparison

This note compares three `MarketDepthProcessor` configurations on the first 5 000 filtered ticks of `glbx-mdp3-20240405.mbp-10.dbn.zst` (AUDUSD micro):

| Variant | Config | Output Shape | Notes |
| --- | --- | --- | --- |
| Baseline 402 | `price_range=200` | `(402, 50)` | Legacy ladder, ±20 pips window |
| Wide → 100 | `price_range=392`, `target_price_levels=100` | `(100, 50)` | Captures ±39.2 pips, pools to 100 rows |
| Wide → 50 | `price_range=384`, `target_price_levels=50` | `(50, 50)` | Captures ±38.4 pips, pools to 50 rows |
| Pip Bins | `price_range=400`, `bin_spec=[(≤10 pips → 1 pip), (≤20 → 2 pips), (>20 → 3 pips)]` | `(46, 50)` | Variable pip-width ladder spanning ±40 pips |

All runs use the enhanced collapse tooling: either a fixed target level count or the new `bin_spec` schedule. The raw ladder is sized by `price_range`, then the processor collapses bid/ask rows into balanced groups that sum the contained liquidity. Each heatmap now overlays the normalized mid price as a black trace so the ladder movement stays aligned to price action.

## Heatmaps

**Volume**
- Baseline output (402×50): ![Baseline Volume](../outputs/market_depth_collapse/baseline_402_volume.png)
- Wide capture collapsed to 100 levels: ![100 Volume](../outputs/market_depth_collapse/wide_to_100_volume.png)
- Wide capture collapsed to 50 levels: ![50 Volume](../outputs/market_depth_collapse/wide_to_50_volume.png)
- Variable pip bins (1→2→3 pip widths): ![Pip Volume](../outputs/market_depth_collapse/pip_bins_variable_volume.png)

**Trade Counts**
- Baseline counts: ![Baseline Counts](../outputs/market_depth_collapse/baseline_402_trade_counts.png)
- Wide → 100 counts: ![100 Counts](../outputs/market_depth_collapse/wide_to_100_trade_counts.png)
- Wide → 50 counts: ![50 Counts](../outputs/market_depth_collapse/wide_to_50_trade_counts.png)
- Pip-bin counts: ![Pip Counts](../outputs/market_depth_collapse/pip_bins_variable_trade_counts.png)

**Variance**
- Baseline variance: ![Baseline Variance](../outputs/market_depth_collapse/baseline_402_variance.png)
- Wide → 100 variance: ![100 Variance](../outputs/market_depth_collapse/wide_to_100_variance.png)
- Wide → 50 variance: ![50 Variance](../outputs/market_depth_collapse/wide_to_50_variance.png)
- Pip-bin variance: ![Pip Variance](../outputs/market_depth_collapse/pip_bins_variable_variance.png)

## Differences Versus Baseline

To highlight redistribution after pooling, the baseline tensor was recomputed with the same `price_range` and re-pooled using the exact groupings before subtraction.

- 100-level collapse vs baseline: ![Diff 100](../outputs/market_depth_collapse/diff_wide_to_100.png)
- 50-level collapse vs baseline: ![Diff 50](../outputs/market_depth_collapse/diff_wide_to_50.png)
- Pip bins vs wide baseline: ![Diff Pip](../outputs/market_depth_collapse/diff_pip_bins_variable.png)

The collapses retain high-intensity regions while adding coverage deeper in the book, allowing downstream models to trade resolution for range without losing aggregate volume.

## Representation Trade-Offs for Forecasting

- **Baseline 402** – Maximum micro-level fidelity, but the ±20 pip window limits awareness of
  liquidity shifts building just outside the ladder; best when microstructure detail outweighs
  range.
- **Wide → 100** – Balanced compromise; twice the spatial reach with moderate pooling that keeps
  the central curvature intact. Works well for medium-horizon behaviour models that still need
  orderly gradients.
- **Wide → 50** – Heavy compression; exposes trend in deep liquidity but blurs near-touch cues.
  Useful for downstream factor inputs, less so for real-time signal generation.
- **Pip Bins** – Variable-width bins preserve single-pip resolution near the mid while stretching to
  ±40 pips. This mirrors the paper’s market-depth formulation and provides the richest picture for
  behaviour prediction models that must reason about both queue dynamics and deeper liquidity
  shocks.

**Recommendation:** For market-behaviour forecasting, the pip-bin representation offers the
strongest blend of microscopic resolution and stable depth context. It gives learning models the
most information to anticipate regime shifts without being overwhelmed by redundant near-mid rows.

## Variant Pros and Cons

- **Baseline 402**
  - *Pros*: Full one-pip granularity across the legacy ±20 pip ladder; minimal preprocessing, so it
    matches historical pipelines and keeps the original signal intact for microstructure studies.
  - *Cons*: Narrow spatial field misses liquidity rebalancing outside the top 20 pips and drives up
    compute for models that only need coarse depth changes.
- **Wide → 100**
  - *Pros*: Captures the broader ±39.2 pip context while halving the ladder height; pooling keeps
    gradients smooth enough for CNN-style models and reduces noise from empty levels.
  - *Cons*: Collapsing eight native rows per bin can mute sharp queue imbalances close to the touch
    and introduces stride artefacts if you need single-pip features.
- **Wide → 50**
  - *Pros*: The smallest tensor with wide coverage; great for factor libraries, statistical summaries,
    or latency-sensitive inference where coarse liquidity bands are sufficient.
  - *Cons*: Twenty‑six native rows are aggregated into each bin, so fine-order-book structure and
    spread dynamics are mostly lost.
- **Pip Bins (1 pip → 2 pip → 3 pip)**
  - *Pros*: Retains 1‑pip fidelity around the mid, then widens bins smoothly to ±40 pips; aligns with
    the Wu et al. market-depth representation and remains translation invariant after the per-bin
    centring rework.
  - *Cons*: Non-uniform bins complicate direct linear indexing and require the bin schedule to be
    synchronised across any downstream features.

## Variance and Trade Count Behaviour

- **Baseline 402**
  - Variance reflects subtle volatility differences at each single pip; trade counts remain sparse
    but precise, capturing queue churn right at the touch.
- **Wide → 100**
  - Averaging eight rows per bin smooths the variance surface and highlights medium-depth bursts; trade
    counts are aggregated consistently with volume (verified via `bin_spec` tests), so activity patterns
    remain aligned with the pooled ladder.
- **Wide → 50**
  - Variance becomes a coarse volatility envelope—useful for regime labelling but too blunt for
    latency trading. Count spikes blend together, which can hide short-lived sweeps.
- **Pip Bins**
  - Near the mid, variance and counts behave like the baseline (1 pip bins); beyond ±10 pips the wider
    bins average activity, giving a clearer picture of deeper liquidity churn without overwhelming the
    tensor. The shared bin mapping ensures trade-count features stay in lockstep with the volume-based
    aggregation schedule.
