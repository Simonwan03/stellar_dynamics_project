# Task E summary: `tcn`

## E1 one-shot 512 -> 128

- MSE (z): 0.6836599631853307
- MAE (z): 0.6585103298502679
- RMSE (z): 0.8268373257088305
- Pearson: 0.5881708308684533
- Unit-circle error: 0.786338819041661
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 0.6836599631853307, 'ci95_low': 0.6704966565494871, 'ci95_high': 0.6969674257609226, 'num_systems': 652}, 'mae_z': {'mean': 0.6585103298502679, 'ci95_low': 0.6517561041505305, 'ci95_high': 0.6655855569754172, 'num_systems': 652}, 'rmse_z': {'mean': 0.8268373257088305, 'ci95_low': 0.8188386022418027, 'ci95_high': 0.8348457492378645, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 0, 'train_windows_per_trajectory': 8, 'val_windows_per_trajectory': 4, 'test_windows_per_trajectory': 4, 'rolling_windows_per_trajectory': 2, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system', 'last_train_epoch': 10, 'last_train_epoch_windows': 62375}
