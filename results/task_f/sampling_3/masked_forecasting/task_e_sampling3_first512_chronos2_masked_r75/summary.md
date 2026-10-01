# Task E summary: `chronos2`

## E1 one-shot 512 -> 128

- MSE (z): 1.058969411628683
- MAE (z): 0.890257928433022
- RMSE (z): 1.0290623944293578
- Pearson: 0.4979629463394199
- Unit-circle error: 0.9997258648933682
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 1.058969411628683, 'ci95_low': 1.0379135530675196, 'ci95_high': 1.0792687584868736, 'num_systems': 652}, 'mae_z': {'mean': 0.890257928433022, 'ci95_low': 0.8823033628308699, 'ci95_high': 0.8979574643932522, 'num_systems': 652}, 'rmse_z': {'mean': 1.0290623944293578, 'ci95_low': 1.0187804238591471, 'ci95_high': 1.0388786062169753, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 0, 'train_windows_per_trajectory': 8, 'val_windows_per_trajectory': 4, 'test_windows_per_trajectory': 4, 'rolling_windows_per_trajectory': 2, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system'}
