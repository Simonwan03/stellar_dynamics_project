# Task E summary: `linear_extrapolation`

## E1 one-shot 512 -> 128

- MSE (z): 1.4661618961184189
- MAE (z): 0.5739736049042794
- RMSE (z): 1.2108517234238132
- Pearson: 0.5847835823059925
- Unit-circle error: 4.4005202532742974e-08
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 1.4661618961184189, 'ci95_low': 1.3643187948655275, 'ci95_high': 1.5608809824928453, 'num_systems': 652}, 'mae_z': {'mean': 0.5739736049042794, 'ci95_low': 0.5539752311239311, 'ci95_high': 0.5946444800752297, 'num_systems': 652}, 'rmse_z': {'mean': 1.2108517234238132, 'ci95_low': 1.1680405792787445, 'ci95_high': 1.2493522249955302, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 12535, 'train_windows_per_trajectory': 1, 'val_windows_per_trajectory': 1, 'test_windows_per_trajectory': 1, 'rolling_windows_per_trajectory': 1, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system'}

## E2 recursive rolling

- Windows: 12535
- Horizon: 512
- Block 1 MAE: 0.5791150973927678
- Block 2 MAE: 1.0221181305519353
- Block 3 MAE: 1.4237204750132408
- Block 4 MAE: 1.8171689796055428
- Error amplification ratio: 3.137837344919192
