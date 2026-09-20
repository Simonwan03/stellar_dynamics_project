# Task E summary: `tcn`

## E1 one-shot 512 -> 128

- MSE (z): 0.3721960777799565
- MAE (z): 0.41251369816753974
- RMSE (z): 0.6100787471957669
- Pearson: 0.7732481679095434
- Unit-circle error: 0.589303728786698
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 0.3721960777799565, 'ci95_low': 0.361393832151679, 'ci95_high': 0.38323446261760025, 'num_systems': 652}, 'mae_z': {'mean': 0.41251369816753974, 'ci95_low': 0.40527366662968756, 'ci95_high': 0.4198001122254421, 'num_systems': 652}, 'rmse_z': {'mean': 0.6100787471957669, 'ci95_low': 0.6011604043505167, 'ci95_high': 0.61905933688805, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 0, 'train_windows_per_trajectory': 8, 'val_windows_per_trajectory': 4, 'test_windows_per_trajectory': 4, 'rolling_windows_per_trajectory': 2, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system'}
