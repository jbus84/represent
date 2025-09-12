# Symbol-Specific Parameter Optimization Results

Generated on: 2025-09-12 07:29:50

## Summary
- **Total symbols optimized**: 6
- **Methods used**: triple_exceedance, triple_barrier, oracle_ternary, oracle_binary, binary_ctl, ternary_ctl
- **Total optimization runs**: 36

## Returns Performance
- **Best performing method**: triple_exceedance (52.6612)
- **Average returns**: 6.7301
- **Returns range**: 0.0146 to 52.6612

## Parameter Comparison Table

| symbol                                    | method            | timestamp                  |   lookforward_window |   scaling_factor |   returns |   dataset_size |   sample_efficiency |   barrier_width |   transaction_cost |   neutral_reward_factor |     omega |   marginal_change_thres |   window_size |
|:------------------------------------------|:------------------|:---------------------------|---------------------:|-----------------:|----------:|---------------:|--------------------:|----------------:|-------------------:|------------------------:|----------:|------------------------:|--------------:|
| M6AZ4_inputs_only_dataset_20250909_141012 | triple_exceedance | 2025-09-12T07:29:23.271374 |                 1624 |          2.77997 | 24.1081   |       26441376 |             23.6372 |      nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | triple_barrier    | 2025-09-12T07:29:23.271728 |                 2722 |        nan       |  5.02676  |       26441376 |             23.6372 |        0.000477 |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | oracle_ternary    | 2025-09-12T07:29:23.271060 |                  nan |        nan       |  0.043965 |       26441376 |             23.6372 |      nan        |            8.5e-05 |                0.384936 | nan       |                 nan     |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | oracle_binary     | 2025-09-12T07:29:23.270663 |                  nan |        nan       |  0.042856 |       26441376 |             23.6372 |      nan        |            9.8e-05 |              nan        | nan       |                 nan     |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | binary_ctl        | 2025-09-12T07:29:23.267779 |                  nan |        nan       |  0.143043 |       26441376 |             23.6372 |      nan        |          nan       |              nan        |   2.6e-05 |                 nan     |           nan |
| M6AZ4_inputs_only_dataset_20250909_141012 | ternary_ctl       | 2025-09-12T07:29:23.270300 |                  nan |        nan       |  0.393798 |       26441376 |             23.6372 |      nan        |          nan       |              nan        | nan       |                   1e-05 |            66 |
| M6AM5_inputs_only_dataset_20250909_140854 | triple_exceedance | 2025-09-12T07:22:13.808464 |                 1727 |          2.91702 | 52.6612   |        3530614 |            177.023  |      nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | triple_barrier    | 2025-09-12T07:22:13.808624 |                 2104 |        nan       |  8.83996  |        3530614 |            177.023  |        0.000743 |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | oracle_ternary    | 2025-09-12T07:22:13.808264 |                  nan |        nan       |  0.11043  |        3530614 |            177.023  |      nan        |            7.2e-05 |                0.372918 | nan       |                 nan     |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | oracle_binary     | 2025-09-12T07:22:13.807944 |                  nan |        nan       |  0.113201 |        3530614 |            177.023  |      nan        |            7.3e-05 |              nan        | nan       |                 nan     |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | binary_ctl        | 2025-09-12T07:22:13.805915 |                  nan |        nan       |  0.17082  |        3530614 |            177.023  |      nan        |          nan       |              nan        |   1.8e-05 |                 nan     |           nan |
| M6AM5_inputs_only_dataset_20250909_140854 | ternary_ctl       | 2025-09-12T07:22:13.807569 |                  nan |        nan       |  0.474257 |        3530614 |            177.023  |      nan        |          nan       |              nan        | nan       |                   5e-06 |             6 |
| M6AH5_inputs_only_dataset_20250909_140842 | triple_exceedance | 2025-09-12T07:27:28.496379 |                 1383 |          4.45709 | 27.9032   |       18979389 |             32.9305 |      nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | triple_barrier    | 2025-09-12T07:27:28.496534 |                 2917 |        nan       |  5.43338  |       18979389 |             32.9305 |        0.000429 |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | oracle_ternary    | 2025-09-12T07:27:28.496221 |                  nan |        nan       |  0.119567 |       18979389 |             32.9305 |      nan        |            2.3e-05 |                0.366549 | nan       |                 nan     |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | oracle_binary     | 2025-09-12T07:27:28.496060 |                  nan |        nan       |  0.057879 |       18979389 |             32.9305 |      nan        |            6.3e-05 |              nan        | nan       |                 nan     |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | binary_ctl        | 2025-09-12T07:27:28.493895 |                  nan |        nan       |  0.10559  |       18979389 |             32.9305 |      nan        |          nan       |              nan        |   5.4e-05 |                 nan     |           nan |
| M6AH5_inputs_only_dataset_20250909_140842 | ternary_ctl       | 2025-09-12T07:27:28.495870 |                  nan |        nan       |  1.12362  |       18979389 |             32.9305 |      nan        |          nan       |              nan        | nan       |                   4e-06 |            55 |
| M6AU5_inputs_only_dataset_20250909_140950 | triple_exceedance | 2025-09-12T07:25:37.026108 |                 1218 |          5.13014 | 52.5038   |         102043 |           6124.87   |      nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | triple_barrier    | 2025-09-12T07:25:37.027126 |                 2377 |        nan       | 16.345    |         102043 |           6124.87   |        0.001    |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | oracle_ternary    | 2025-09-12T07:25:37.025698 |                  nan |        nan       |  0.222085 |         102043 |           6124.87   |      nan        |            6.3e-05 |                0.300807 | nan       |                 nan     |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | oracle_binary     | 2025-09-12T07:25:37.025424 |                  nan |        nan       |  0.17171  |         102043 |           6124.87   |      nan        |            4.4e-05 |              nan        | nan       |                 nan     |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | binary_ctl        | 2025-09-12T07:25:37.023673 |                  nan |        nan       |  0.332965 |         102043 |           6124.87   |      nan        |          nan       |              nan        |   3.7e-05 |                 nan     |           nan |
| M6AU5_inputs_only_dataset_20250909_140950 | ternary_ctl       | 2025-09-12T07:25:37.025229 |                  nan |        nan       |  1.00087  |         102043 |           6124.87   |      nan        |          nan       |              nan        | nan       |                   4e-06 |            96 |
| M6AU4_inputs_only_dataset_20250909_140914 | triple_exceedance | 2025-09-12T07:20:27.183348 |                 2031 |          3.38972 | 18.0668   |       21854618 |             28.5981 |      nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | triple_barrier    | 2025-09-12T07:20:27.183745 |                 2978 |        nan       |  5.00061  |       21854618 |             28.5981 |        0.00038  |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | oracle_ternary    | 2025-09-12T07:20:27.182951 |                  nan |        nan       |  0.020573 |       21854618 |             28.5981 |      nan        |            7.9e-05 |                0.339519 | nan       |                 nan     |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | oracle_binary     | 2025-09-12T07:20:27.182784 |                  nan |        nan       |  0.041397 |       21854618 |             28.5981 |      nan        |            9.7e-05 |              nan        | nan       |                 nan     |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | binary_ctl        | 2025-09-12T07:20:27.178401 |                  nan |        nan       |  0.055598 |       21854618 |             28.5981 |      nan        |          nan       |              nan        |   3.6e-05 |                 nan     |           nan |
| M6AU4_inputs_only_dataset_20250909_140914 | ternary_ctl       | 2025-09-12T07:20:27.182331 |                  nan |        nan       |  0.176652 |       21854618 |             28.5981 |      nan        |          nan       |              nan        | nan       |                   7e-06 |            18 |
| M6AM4_inputs_only_dataset_20250909_140944 | triple_exceedance | 2025-09-12T07:24:00.782694 |                 3370 |          2.23225 | 17.6742   |       15089079 |             41.4207 |      nan        |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | triple_barrier    | 2025-09-12T07:24:00.782882 |                 2992 |        nan       |  3.59805  |       15089079 |             41.4207 |        0.000478 |          nan       |              nan        | nan       |                 nan     |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | oracle_ternary    | 2025-09-12T07:24:00.782480 |                  nan |        nan       |  0.014643 |       15089079 |             41.4207 |      nan        |            7.9e-05 |                0.30402  | nan       |                 nan     |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | oracle_binary     | 2025-09-12T07:24:00.782204 |                  nan |        nan       |  0.030916 |       15089079 |             41.4207 |      nan        |            8.5e-05 |              nan        | nan       |                 nan     |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | binary_ctl        | 2025-09-12T07:24:00.779952 |                  nan |        nan       |  0.030586 |       15089079 |             41.4207 |      nan        |          nan       |              nan        |   4.9e-05 |                 nan     |           nan |
| M6AM4_inputs_only_dataset_20250909_140944 | ternary_ctl       | 2025-09-12T07:24:00.781808 |                  nan |        nan       |  0.125792 |       15089079 |             41.4207 |      nan        |          nan       |              nan        | nan       |                   5e-06 |            78 |

## Triple Exceedance Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 52.6612 (M6AM5_inputs_only_dataset_20250909_140854)
- **Average returns**: 32.1529
- **Worst returns**: 17.6742 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **lookforward_window**: 1218 to 3370 (avg: 1892.17)
- **scaling_factor**: 2.23225 to 5.13014 (avg: 3.48437)

## Triple Barrier Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 16.3450 (M6AU5_inputs_only_dataset_20250909_140950)
- **Average returns**: 7.3740
- **Worst returns**: 3.5980 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **lookforward_window**: 2104 to 2992 (avg: 2681.67)
- **barrier_width**: 0.000380057 to 0.000999514 (avg: 0.000584331)

## Oracle Ternary Results

Optimized for **6** symbols.

### Performance Metrics
- **Best returns**: 0.2221 (M6AU5_inputs_only_dataset_20250909_140950)
- **Average returns**: 0.0885
- **Worst returns**: 0.0146 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **transaction_cost**: 2.29487e-05 to 8.49198e-05 (avg: 6.69557e-05)
- **neutral_reward_factor**: 0.300807 to 0.384936 (avg: 0.344791)

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
- **Best returns**: 1.1236 (M6AH5_inputs_only_dataset_20250909_140842)
- **Average returns**: 0.5492
- **Worst returns**: 0.1258 (M6AM4_inputs_only_dataset_20250909_140944)

### Parameter Ranges
- **marginal_change_thres**: 3.73818e-06 to 9.94289e-06 (avg: 5.82273e-06)
- **window_size**: 6 to 96 (avg: 53.1667)

---
*Generated by Represent Parameter Optimization System*