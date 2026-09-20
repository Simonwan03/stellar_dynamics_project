# Task E summary: `tirex2`

## E1 one-shot 512 -> 128

- MSE (z): 0.49092648176719766
- MAE (z): 0.5870552842529643
- RMSE (z): 0.700661460169744
- Pearson: 0.8320262607621245
- Unit-circle error: 0.8436800746393881
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 0.49092648176719766, 'ci95_low': 0.48179208869578116, 'ci95_high': 0.5003184394600299, 'num_systems': 652}, 'mae_z': {'mean': 0.5870552842529643, 'ci95_low': 0.5815886011977176, 'ci95_high': 0.5923406855080579, 'num_systems': 652}, 'rmse_z': {'mean': 0.700661460169744, 'ci95_low': 0.6941124464921647, 'ci95_high': 0.7073319160451801, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 0, 'train_windows_per_trajectory': 8, 'val_windows_per_trajectory': 4, 'test_windows_per_trajectory': 4, 'rolling_windows_per_trajectory': 2, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system'}
