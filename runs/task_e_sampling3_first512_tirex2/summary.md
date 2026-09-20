# Task E summary: `tirex2`

## E1 one-shot 512 -> 128

- MSE (z): 0.2374177138653798
- MAE (z): 0.2325835683619211
- RMSE (z): 0.4872552861338498
- Pearson: 0.8632158196493117
- Unit-circle error: 0.17322379403497937
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 0.2374177138653798, 'ci95_low': 0.223704672352275, 'ci95_high': 0.25187433711919843, 'num_systems': 652}, 'mae_z': {'mean': 0.2325835683619211, 'ci95_low': 0.22283609815596636, 'ci95_high': 0.2435525645702389, 'num_systems': 652}, 'rmse_z': {'mean': 0.4872552861338498, 'ci95_low': 0.4729742799189539, 'ci95_high': 0.501870836119902, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 12535, 'train_windows_per_trajectory': 1, 'val_windows_per_trajectory': 1, 'test_windows_per_trajectory': 1, 'rolling_windows_per_trajectory': 1, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system'}

## E2 recursive rolling

- Windows: 12535
- Horizon: 512
- Block 1 MAE: 0.2275470006863262
- Block 2 MAE: 0.3339593748788607
- Block 3 MAE: 0.37816465774046754
- Block 4 MAE: 0.4074778281030293
- Error amplification ratio: 1.7907413715583886
