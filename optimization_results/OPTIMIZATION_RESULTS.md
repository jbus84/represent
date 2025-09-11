# Symbol-Specific Parameter Optimization Results

Generated on: 2025-09-11 19:21:29

## Summary
- **Total symbols optimized**: 6
- **Methods used**: triple_exceedance, triple_barrier, oracle_ternary, oracle_binary, binary_ctl, ternary_ctl
- **Total optimization runs**: 36

## Returns Performance
- **Best performing method**: ternary_ctl (0.8557)
- **Average returns**: 0.1055
- **Returns range**: -0.2089 to 0.8557

## Parameter Comparison Table

| symbol                                    | method            | timestamp                  |   lookforward_window |   scaling_factor |   min_exceedance_threshold |   volatility_window |   window_penalty_weight |   balance_weight |   target_balance_ratio |   adaptive_scaling |   returns |   dataset_size |   sample_efficiency |   barrier_width |   min_return_threshold |   normalize_by_volatility |   transaction_cost |   neutral_reward_factor |     omega |   marginal_change_thres |   window_size |
|:------------------------------------------|:------------------|:---------------------------|---------------------:|-----------------:|---------------------------:|--------------------:|------------------------:|-----------------:|-----------------------:|-------------------:|----------:|---------------:|--------------------:|----------------:|-----------------------:|--------------------------:|-------------------:|------------------------:|----------:|------------------------:|--------------:|
| M6AZ4_inputs_only_dataset_20250909_141012 | triple_exceedance | 2025-09-11T19:20:52.375996 |                 2395 |          3.48428 |                   0.378142 |             26.4526 |                0.276929 |         0.489633 |               0.318612 |           0.286231 | -0.006354 |       26441376 |             23.6372 |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | triple_barrier    | 2025-09-11T19:20:52.375823 |                 1419 |        nan       |                 nan        |             43.036  |              nan        |       nan        |             nan        |         nan        | -0.022087 |       26441376 |             23.6372 |        0.00016  |                2e-05   |                  0.482849 |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | oracle_ternary    | 2025-09-11T19:20:52.375666 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.049341 |       26441376 |             23.6372 |      nan        |              nan       |                nan        |            8.5e-05 |                0.384936 | nan       |                 nan     |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | oracle_binary     | 2025-09-11T19:20:52.375462 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.042856 |       26441376 |             23.6372 |      nan        |              nan       |                nan        |            9.8e-05 |              nan        | nan       |                 nan     |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | binary_ctl        | 2025-09-11T19:20:52.374846 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.143043 |       26441376 |             23.6372 |      nan        |              nan       |                nan        |          nan       |              nan        |   2.6e-05 |                 nan     |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | ternary_ctl       | 2025-09-11T19:20:52.375267 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.328803 |       26441376 |             23.6372 |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                   9e-06 |            62 |
| M6AM5_inputs_only_dataset_20250909_140854 | triple_exceedance | 2025-09-11T19:09:21.557769 |                 2854 |          2.29295 |                   0.201919 |             40.5529 |                0.228271 |         0.244208 |               0.372628 |           0.406153 | -0.006625 |        3530614 |            177.023  |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | triple_barrier    | 2025-09-11T19:09:21.557591 |                 1388 |        nan       |                 nan        |             27.3844 |              nan        |       nan        |             nan        |         nan        | -0.05881  |        3530614 |            177.023  |        0.000169 |                2.7e-05 |                  0.187061 |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | oracle_ternary    | 2025-09-11T19:09:21.557442 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.120893 |        3530614 |            177.023  |      nan        |              nan       |                nan        |            8e-05   |                0.355104 | nan       |                 nan     |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | oracle_binary     | 2025-09-11T19:09:21.557273 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.113201 |        3530614 |            177.023  |      nan        |              nan       |                nan        |            7.3e-05 |              nan        | nan       |                 nan     |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | binary_ctl        | 2025-09-11T19:09:21.555884 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.17082  |        3530614 |            177.023  |      nan        |              nan       |                nan        |          nan       |              nan        |   1.8e-05 |                 nan     |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | ternary_ctl       | 2025-09-11T19:09:21.557061 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.308769 |        3530614 |            177.023  |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                   8e-06 |             6 |
| M6AH5_inputs_only_dataset_20250909_140842 | triple_exceedance | 2025-09-11T19:18:05.432559 |                 2726 |          3.4436  |                   0.448674 |             26.1868 |                0.265838 |         0.368562 |               0.262432 |           0.326416 | -0.007136 |       18979389 |             32.9305 |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | triple_barrier    | 2025-09-11T19:18:05.432388 |                 2076 |        nan       |                 nan        |             39.205  |              nan        |       nan        |             nan        |         nan        | -0.02805  |       18979389 |             32.9305 |        0.000199 |                2.1e-05 |                  0.449262 |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | oracle_ternary    | 2025-09-11T19:18:05.432215 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.127339 |       18979389 |             32.9305 |      nan        |              nan       |                nan        |            2.3e-05 |                0.366549 | nan       |                 nan     |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | oracle_binary     | 2025-09-11T19:18:05.431799 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.057879 |       18979389 |             32.9305 |      nan        |              nan       |                nan        |            6.3e-05 |              nan        | nan       |                 nan     |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | binary_ctl        | 2025-09-11T19:18:05.430650 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.10559  |       18979389 |             32.9305 |      nan        |              nan       |                nan        |          nan       |              nan        |   5.4e-05 |                 nan     |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | ternary_ctl       | 2025-09-11T19:18:05.431549 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.855718 |       18979389 |             32.9305 |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                   4e-06 |            55 |
| M6AU5_inputs_only_dataset_20250909_140950 | triple_exceedance | 2025-09-11T19:15:05.778135 |                 2987 |          2.26213 |                   0.324825 |             34.0109 |                0.224994 |         0.446245 |               0.38237  |           0.732322 | -0.008649 |         102043 |           6124.87   |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | triple_barrier    | 2025-09-11T19:15:05.777964 |                 2994 |        nan       |                 nan        |             48.0706 |              nan        |       nan        |             nan        |         nan        | -0.208946 |         102043 |           6124.87   |        0.000102 |                1e-05   |                  0.332493 |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | oracle_ternary    | 2025-09-11T19:15:05.777675 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.240254 |         102043 |           6124.87   |      nan        |              nan       |                nan        |            6.3e-05 |                0.300807 | nan       |                 nan     |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | oracle_binary     | 2025-09-11T19:15:05.777508 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.17171  |         102043 |           6124.87   |      nan        |              nan       |                nan        |            4.4e-05 |              nan        | nan       |                 nan     |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | binary_ctl        | 2025-09-11T19:15:05.776061 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.332965 |         102043 |           6124.87   |      nan        |              nan       |                nan        |          nan       |              nan        |   3.7e-05 |                 nan     |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | ternary_ctl       | 2025-09-11T19:15:05.777309 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.667979 |         102043 |           6124.87   |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                   6e-06 |            21 |
| M6AU4_inputs_only_dataset_20250909_140914 | triple_exceedance | 2025-09-11T19:06:39.300383 |                 2710 |          3.52158 |                   0.324283 |             19.6709 |                0.164634 |         0.739057 |               0.353855 |           0.394424 | -0.004707 |       21854618 |             28.5981 |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | triple_barrier    | 2025-09-11T19:06:39.299924 |                 1040 |        nan       |                 nan        |             18.2709 |              nan        |       nan        |             nan        |         nan        | -0.027569 |       21854618 |             28.5981 |        0.000173 |                3e-05   |                  0.144091 |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | oracle_ternary    | 2025-09-11T19:06:39.299743 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.025847 |       21854618 |             28.5981 |      nan        |              nan       |                nan        |            7.9e-05 |                0.339519 | nan       |                 nan     |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | oracle_binary     | 2025-09-11T19:06:39.299404 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.041397 |       21854618 |             28.5981 |      nan        |              nan       |                nan        |            9.7e-05 |              nan        | nan       |                 nan     |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | binary_ctl        | 2025-09-11T19:06:39.297483 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.055598 |       21854618 |             28.5981 |      nan        |              nan       |                nan        |          nan       |              nan        |   3.6e-05 |                 nan     |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | ternary_ctl       | 2025-09-11T19:06:39.299202 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.105801 |       21854618 |             28.5981 |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                   7e-06 |            18 |
| M6AM4_inputs_only_dataset_20250909_140944 | triple_exceedance | 2025-09-11T19:12:22.569700 |                 2202 |          3.41615 |                   0.108234 |             48.7964 |                0.266489 |         0.327403 |               0.277274 |           0.183405 | -0.007562 |       15089079 |             41.4207 |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | triple_barrier    | 2025-09-11T19:12:22.569533 |                 1221 |        nan       |                 nan        |             16.78   |              nan        |       nan        |             nan        |         nan        | -0.024622 |       15089079 |             41.4207 |        0.00015  |                2.2e-05 |                  0.107161 |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | oracle_ternary    | 2025-09-11T19:12:22.569379 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.01929  |       15089079 |             41.4207 |      nan        |              nan       |                nan        |            7.1e-05 |                0.30222  | nan       |                 nan     |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | oracle_binary     | 2025-09-11T19:12:22.569210 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.030916 |       15089079 |             41.4207 |      nan        |              nan       |                nan        |            8.5e-05 |              nan        | nan       |                 nan     |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | binary_ctl        | 2025-09-11T19:12:22.567716 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.030586 |       15089079 |             41.4207 |      nan        |              nan       |                nan        |          nan       |              nan        |   4.9e-05 |                 nan     |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | ternary_ctl       | 2025-09-11T19:12:22.569011 |                  nan |        nan       |                 nan        |            nan      |              nan        |       nan        |             nan        |         nan        |  0.061638 |       15089079 |             41.4207 |      nan        |              nan       |                nan        |          nan       |              nan        | nan       |                   5e-06 |            35 |

## Triple Exceedance Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: -0.0047 (M6AU4_inputs_only_dataset_20250909_140914)
- **Average returns**: -0.0068
- **Worst returns**: -0.0086 (M6AU5_inputs_only_dataset_20250909_140950)

### Parameter Ranges
- **lookforward_window**: 2202 to 2987 (avg: 2645.67)
- **scaling_factor**: 2.26213 to 3.52158 (avg: 3.07011)


## Triple Barrier Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: -0.0221 (M6AZ4_inputs_only_dataset_20250909_141012)
- **Average returns**: -0.0617
- **Worst returns**: -0.2089 (M6AU5_inputs_only_dataset_20250909_140950)

### Parameter Ranges
- **lookforward_window**: 1040 to 2994 (avg: 1689.67)
- **barrier_width**: 0.000101774 to 0.000199227 (avg: 0.000158916)

## Oracle Ternary Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 0.2403 (M6AU5_inputs_only_dataset_20250909_140950)
- **Average returns**: 0.0972
- **Worst returns**: 0.0193 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **transaction_cost**: 2.29487e-05 to 8.49198e-05 (avg: 6.67843e-05)
- **neutral_reward_factor**: 0.300807 to 0.384936 (avg: 0.341522)

## Oracle Binary Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 0.1717 (M6AU5_inputs_only_dataset_20250909_140950)
- **Average returns**: 0.0763
- **Worst returns**: 0.0309 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **transaction_cost**: 4.37086e-05 to 9.75777e-05 (avg: 7.65302e-05)

## Binary Ctl Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 0.3330 (M6AU5_inputs_only_dataset_20250909_140950)
- **Average returns**: 0.1398
- **Worst returns**: 0.0306 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **omega**: 1.80217e-05 to 5.38133e-05 (avg: 3.67913e-05)

## Ternary Ctl Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 0.8557 (M6AH5_inputs_only_dataset_20250909_140842)
- **Average returns**: 0.3881
- **Worst returns**: 0.0616 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **marginal_change_thres**: 3.73818e-06 to 9.2268e-06 (avg: 6.38014e-06)
- **window_size**: 6 to 62 (avg: 32.8333)

---
*Generated by Represent Parameter Optimization System*