# Task E summary: `toto2`

## E1 one-shot 512 -> 128

- MSE (z): 0.21765900172013783
- MAE (z): 0.22229725372884168
- RMSE (z): 0.4665393892482583
- Pearson: 0.8736700448453399
- Unit-circle error: 0.1820412776709987
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 0.21765900172013783, 'ci95_low': 0.2050484317146676, 'ci95_high': 0.23095101508674096, 'num_systems': 652}, 'mae_z': {'mean': 0.22229725372884168, 'ci95_low': 0.21278635362127687, 'ci95_high': 0.23283849764580217, 'num_systems': 652}, 'rmse_z': {'mean': 0.4665393892482583, 'ci95_low': 0.4528227372522274, 'ci95_high': 0.48057363115208107, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 12535, 'train_windows_per_trajectory': 1, 'val_windows_per_trajectory': 1, 'test_windows_per_trajectory': 1, 'rolling_windows_per_trajectory': 1, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system'}

## E2 recursive rolling

- Windows: 12535
- Horizon: 512
- Block 1 MAE: 0.21627882380751964
- Block 2 MAE: 0.31950717249042937
- Block 3 MAE: 0.36514449416749567
- Block 4 MAE: 0.3998385682366002
- Error amplification ratio: 1.8487180630889797
