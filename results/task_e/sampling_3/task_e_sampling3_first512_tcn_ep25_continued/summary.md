# Task E summary: `tcn`

## E1 one-shot 512 -> 128

- MSE (z): 0.3475369955139815
- MAE (z): 0.38284858112001635
- RMSE (z): 0.5895226844778592
- Pearson: 0.7878127837805358
- Unit-circle error: 0.5474906925103237
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 0.3475369955139815, 'ci95_low': 0.3365693901949824, 'ci95_high': 0.35822676237284773, 'num_systems': 652}, 'mae_z': {'mean': 0.38284858112001635, 'ci95_low': 0.37481117352112536, 'ci95_high': 0.39042028272966034, 'num_systems': 652}, 'rmse_z': {'mean': 0.5895226844778592, 'ci95_low': 0.5801460076520889, 'ci95_high': 0.5985204768603741, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 12535, 'train_windows_per_trajectory': 1, 'val_windows_per_trajectory': 1, 'test_windows_per_trajectory': 1, 'rolling_windows_per_trajectory': 1, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system', 'last_train_epoch': 25, 'last_train_epoch_windows': 62375}

## E2 recursive rolling

- Windows: 12535
- Horizon: 512
- Block 1 MAE: 0.3837753712729477
- Block 2 MAE: 0.4574634974882174
- Block 3 MAE: 0.5017560503539522
- Block 4 MAE: 0.5545800878677571
- Error amplification ratio: 1.4450642990149822
