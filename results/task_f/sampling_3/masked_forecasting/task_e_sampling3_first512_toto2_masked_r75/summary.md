# Task E summary: `toto2`

## E1 one-shot 512 -> 128

- MSE (z): 1.0660928287416012
- MAE (z): 0.893285824687096
- RMSE (z): 1.032517713524374
- Pearson: 0.26837443348928774
- Unit-circle error: 0.9996678986375693
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 1.0660928287416012, 'ci95_low': 1.0445350469986625, 'ci95_high': 1.0866662690318505, 'num_systems': 652}, 'mae_z': {'mean': 0.893285824687096, 'ci95_low': 0.8853113579015437, 'ci95_high': 0.9011285731204413, 'num_systems': 652}, 'rmse_z': {'mean': 1.032517713524374, 'ci95_low': 1.0220249736779075, 'ci95_high': 1.0424328606464197, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 0, 'train_windows_per_trajectory': 8, 'val_windows_per_trajectory': 4, 'test_windows_per_trajectory': 4, 'rolling_windows_per_trajectory': 2, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system'}
