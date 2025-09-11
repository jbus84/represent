# Symbol-Specific Parameter Optimization Results

Generated on: 2025-09-11 11:07:52

## Summary
- **Total symbols optimized**: 6
- **Methods used**: triple_exceedance, triple_barrier, oracle_ternary, oracle_binary, binary_ctl, ternary_ctl
- **Total optimization runs**: 36

## Returns Performance
- **Best performing method**: ternary_ctl (1.0160)
- **Average returns**: 0.1586
- **Returns range**: -0.2196 to 1.0160

## Parameter Comparison Table

| symbol                                    | method            | timestamp                  |   lookforward_window |   scaling_factor |   min_exceedance_threshold |   volatility_window |   window_penalty_weight |   balance_weight |   target_balance_ratio |   adaptive_scaling |   returns |   dataset_size |   sample_efficiency |   barrier_width |   min_return_threshold |   normalize_by_volatility |   transaction_cost |   neutral_reward_factor |     omega |   marginal_change_thres |   window_size |
|:------------------------------------------|:------------------|:---------------------------|---------------------:|-----------------:|---------------------------:|--------------------:|------------------------:|-----------------:|-----------------------:|-------------------:|----------:|---------------:|--------------------:|----------------:|-----------------------:|--------------------------:|-------------------:|------------------------:|----------:|------------------------:|--------------:|
| M6AZ4_inputs_only_dataset_20250909_141012 | triple_exceedance | 2025-09-11T11:07:46.223430 |                11126 |          7.9343  |                   0.374569 |             680.079 |                0.117822 |         0.570395 |               0.281196 |           0.267884 | -0.039197 |       26441376 |             94.5488 |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |               nan       |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | triple_barrier    | 2025-09-11T11:07:46.223268 |                 7939 |        nan       |                 nan        |             555.053 |              nan        |       nan        |             nan        |         nan        | -0.050826 |       26441376 |             94.5488 |        0.000457 |                  7e-06 |                  0.170144 |          nan       |              nan        | nan       |               nan       |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | oracle_ternary    | 2025-09-11T11:07:46.223085 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.124396 |       26441376 |             94.5488 |      nan        |                nan     |                nan        |            5.8e-05 |                0.342665 | nan       |               nan       |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | oracle_binary     | 2025-09-11T11:07:46.222718 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.176758 |       26441376 |             94.5488 |      nan        |                nan     |                nan        |            8.5e-05 |              nan        | nan       |               nan       |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | binary_ctl        | 2025-09-11T11:07:46.220270 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.29173  |       26441376 |             94.5488 |      nan        |                nan     |                nan        |          nan       |              nan        |   6e-05   |               nan       |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | ternary_ctl       | 2025-09-11T11:07:46.222485 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.33576  |       26441376 |             94.5488 |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |                 1.2e-05 |           240 |
| M6AM5_inputs_only_dataset_20250909_140854 | triple_exceedance | 2025-09-11T10:02:45.934089 |                 6388 |          7.14348 |                   0.355001 |             539.871 |                0.179302 |         0.501843 |               0.281139 |           0.078181 | -0.219637 |        3530614 |            708.092  |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |               nan       |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | triple_barrier    | 2025-09-11T10:02:45.933749 |                 8107 |        nan       |                 nan        |             278.292 |              nan        |       nan        |             nan        |         nan        | -0.206407 |        3530614 |            708.092  |        0.000498 |                  7e-06 |                  0.345952 |          nan       |              nan        | nan       |               nan       |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | oracle_ternary    | 2025-09-11T10:02:45.933426 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.389117 |        3530614 |            708.092  |      nan        |                nan     |                nan        |            2e-05   |                0.315118 | nan       |               nan       |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | oracle_binary     | 2025-09-11T10:02:45.933066 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.374354 |        3530614 |            708.092  |      nan        |                nan     |                nan        |            6e-05   |              nan        | nan       |               nan       |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | binary_ctl        | 2025-09-11T10:02:45.931198 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.492554 |        3530614 |            708.092  |      nan        |                nan     |                nan        |          nan       |              nan        |   3.6e-05 |               nan       |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | ternary_ctl       | 2025-09-11T10:02:45.932863 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  1.01595  |        3530614 |            708.092  |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |                 3.2e-05 |           141 |
| M6AH5_inputs_only_dataset_20250909_140842 | triple_exceedance | 2025-09-11T10:48:36.075502 |                12944 |          7.3642  |                   0.268275 |             724.528 |                0.199283 |         0.75054  |               0.310625 |           0.288547 | -0.054623 |       18979389 |            131.722  |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |               nan       |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | triple_barrier    | 2025-09-11T10:48:36.075246 |                 8745 |        nan       |                 nan        |             678.927 |              nan        |       nan        |             nan        |         nan        | -0.082084 |       18979389 |            131.722  |        0.00048  |                  7e-06 |                  0.156019 |          nan       |              nan        | nan       |               nan       |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | oracle_ternary    | 2025-09-11T10:48:36.075076 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.160285 |       18979389 |            131.722  |      nan        |                nan     |                nan        |            4.4e-05 |                0.301045 | nan       |               nan       |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | oracle_binary     | 2025-09-11T10:48:36.074868 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.216018 |       18979389 |            131.722  |      nan        |                nan     |                nan        |            2.9e-05 |              nan        | nan       |               nan       |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | binary_ctl        | 2025-09-11T10:48:36.073030 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.313138 |       18979389 |            131.722  |      nan        |                nan     |                nan        |          nan       |              nan        |   5.4e-05 |               nan       |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | ternary_ctl       | 2025-09-11T10:48:36.074589 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.404805 |       18979389 |            131.722  |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |                 4.8e-05 |           103 |
| M6AU5_inputs_only_dataset_20250909_140950 | triple_exceedance | 2025-09-11T10:27:47.536722 |                14936 |          7.93793 |                   0.341213 |             604.936 |                0.19893  |         0.712558 |               0.317136 |           0.015018 | -0.04486  |         102043 |          24499.5    |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |               nan       |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | triple_barrier    | 2025-09-11T10:27:47.536558 |                14999 |        nan       |                 nan        |             811.774 |              nan        |       nan        |             nan        |         nan        | -0.057932 |         102043 |          24499.5    |        0.000474 |                  3e-06 |                  0.161445 |          nan       |              nan        | nan       |               nan       |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | oracle_ternary    | 2025-09-11T10:27:47.536411 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.240254 |         102043 |          24499.5    |      nan        |                nan     |                nan        |            6.3e-05 |                0.300807 | nan       |               nan       |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | oracle_binary     | 2025-09-11T10:27:47.536241 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.17171  |         102043 |          24499.5    |      nan        |                nan     |                nan        |            4.4e-05 |              nan        | nan       |               nan       |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | binary_ctl        | 2025-09-11T10:27:47.535031 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.332965 |         102043 |          24499.5    |      nan        |                nan     |                nan        |          nan       |              nan        |   3.7e-05 |               nan       |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | ternary_ctl       | 2025-09-11T10:27:47.536034 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.404123 |         102043 |          24499.5    |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |                 3.2e-05 |           141 |
| M6AU4_inputs_only_dataset_20250909_140914 | triple_exceedance | 2025-09-11T09:44:27.973721 |                14975 |          7.02899 |                   0.508873 |             228.746 |                0.109243 |         0.644415 |               0.340133 |           0.859068 | -0.052041 |       21854618 |            114.392  |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |               nan       |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | triple_barrier    | 2025-09-11T09:44:27.973314 |                 6372 |        nan       |                 nan        |             624.386 |              nan        |       nan        |             nan        |         nan        | -0.06336  |       21854618 |            114.392  |        0.000496 |                  5e-06 |                  0.461695 |          nan       |              nan        | nan       |               nan       |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | oracle_ternary    | 2025-09-11T09:44:27.972797 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.089412 |       21854618 |            114.392  |      nan        |                nan     |                nan        |            6.5e-05 |                0.355798 | nan       |               nan       |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | oracle_binary     | 2025-09-11T09:44:27.972435 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.150933 |       21854618 |            114.392  |      nan        |                nan     |                nan        |            8.7e-05 |              nan        | nan       |               nan       |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | binary_ctl        | 2025-09-11T09:44:27.969512 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.192426 |       21854618 |            114.392  |      nan        |                nan     |                nan        |          nan       |              nan        |   6.3e-05 |               nan       |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | ternary_ctl       | 2025-09-11T09:44:27.971597 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.315732 |       21854618 |            114.392  |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |                 3.2e-05 |           418 |
| M6AM4_inputs_only_dataset_20250909_140944 | triple_exceedance | 2025-09-11T10:24:25.348184 |                11996 |          7.72131 |                   0.412341 |             882.231 |                0.148919 |         0.616184 |               0.380075 |           0.585804 | -0.048821 |       15089079 |            165.683  |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |               nan       |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | triple_barrier    | 2025-09-11T10:24:25.347587 |                 7877 |        nan       |                 nan        |             235.301 |              nan        |       nan        |             nan        |         nan        | -0.044407 |       15089079 |            165.683  |        0.000478 |                  8e-06 |                  0.143999 |          nan       |              nan        | nan       |               nan       |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | oracle_ternary    | 2025-09-11T10:24:25.347436 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.052502 |       15089079 |            165.683  |      nan        |                nan     |                nan        |            6.5e-05 |                0.355798 | nan       |               nan       |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | oracle_binary     | 2025-09-11T10:24:25.347256 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.08505  |       15089079 |            165.683  |      nan        |                nan     |                nan        |            4.4e-05 |              nan        | nan       |               nan       |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | binary_ctl        | 2025-09-11T10:24:25.344505 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.111736 |       15089079 |            165.683  |      nan        |                nan     |                nan        |          nan       |              nan        |   1e-06   |               nan       |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | ternary_ctl       | 2025-09-11T10:24:25.347061 |                  nan |        nan       |                 nan        |             nan     |              nan        |       nan        |             nan        |         nan        |  0.231309 |       15089079 |            165.683  |      nan        |                nan     |                nan        |          nan       |              nan        | nan       |                 4.6e-05 |           304 |

## Triple Exceedance Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: -0.0392 (M6AZ4_inputs_only_dataset_20250909_141012)
- **Average returns**: -0.0765
- **Worst returns**: -0.2196 (M6AM5_inputs_only_dataset_20250909_140854)

### Parameter Ranges
- **lookforward_window**: 6388 to 14975 (avg: 12060.8)
- **scaling_factor**: 7.02899 to 7.93793 (avg: 7.5217)
- **min_exceedance_threshold**: 0.268275 to 0.508873 (avg: 0.376712)
- **volatility_window**: 228.746 to 882.231 (avg: 610.065)
- **window_penalty_weight**: 0.109243 to 0.199283 (avg: 0.158917)
- **balance_weight**: 0.501843 to 0.75054 (avg: 0.632656)
- **target_balance_ratio**: 0.281139 to 0.380075 (avg: 0.318384)
- **adaptive_scaling**: 0.0150178 to 0.859068 (avg: 0.349084)

## Triple Barrier Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: -0.0444 (M6AM4_inputs_only_dataset_20250909_140944)
- **Average returns**: -0.0842
- **Worst returns**: -0.2064 (M6AM5_inputs_only_dataset_20250909_140854)

### Parameter Ranges
- **lookforward_window**: 6372 to 14999 (avg: 9006.5)
- **volatility_window**: 235.301 to 811.774 (avg: 530.622)
- **barrier_width**: 0.000457262 to 0.000497567 (avg: 0.000480349)
- **min_return_threshold**: 3.21682e-06 to 8.48999e-06 (avg: 6.29887e-06)
- **normalize_by_volatility**: 0.143999 to 0.461695 (avg: 0.239876)

## Oracle Ternary Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 0.3891 (M6AM5_inputs_only_dataset_20250909_140854)
- **Average returns**: 0.1760
- **Worst returns**: 0.0525 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **transaction_cost**: 2.01189e-05 to 6.50668e-05 (avg: 5.26379e-05)
- **neutral_reward_factor**: 0.300807 to 0.355798 (avg: 0.328538)

## Oracle Binary Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 0.3744 (M6AM5_inputs_only_dataset_20250909_140854)
- **Average returns**: 0.1958
- **Worst returns**: 0.0850 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **transaction_cost**: 2.90447e-05 to 8.71858e-05 (avg: 5.81205e-05)

## Binary Ctl Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 0.4926 (M6AM5_inputs_only_dataset_20250909_140854)
- **Average returns**: 0.2891
- **Worst returns**: 0.1117 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **omega**: 6.90101e-07 to 6.33929e-05 (avg: 4.18424e-05)

## Ternary Ctl Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 1.0160 (M6AM5_inputs_only_dataset_20250909_140854)
- **Average returns**: 0.4513
- **Worst returns**: 0.2313 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **marginal_change_thres**: 1.20208e-05 to 4.75311e-05 (avg: 3.3422e-05)
- **window_size**: 103 to 418 (avg: 224.5)

---
*Generated by Represent Parameter Optimization System*