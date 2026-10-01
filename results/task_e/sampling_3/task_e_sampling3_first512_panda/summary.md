# Task E summary: `panda_zero_shot`

## E1 one-shot 512 -> 128

- MSE (z): 0.23173840515863312
- MAE (z): 0.2398950345121911
- RMSE (z): 0.48139215319595013
- Pearson: 0.863980570971345
- Unit-circle error: 0.24469884707997824
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 0.23173840515863312, 'ci95_low': 0.2178212343857282, 'ci95_high': 0.24612647193936965, 'num_systems': 652}, 'mae_z': {'mean': 0.2398950345121911, 'ci95_low': 0.22869103904322088, 'ci95_high': 0.2524184669841017, 'num_systems': 652}, 'rmse_z': {'mean': 0.48139215319595013, 'ci95_low': 0.46671321924656806, 'ci95_high': 0.49611135010153995, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 12535, 'train_windows_per_trajectory': 1, 'val_windows_per_trajectory': 1, 'test_windows_per_trajectory': 1, 'rolling_windows_per_trajectory': 1, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system'}

## E2 recursive rolling

- Windows: 12535
- Horizon: 512
- Block 1 MAE: 0.2310641477543618
- Block 2 MAE: 0.3471733448415752
- Block 3 MAE: 0.3821018079844052
- Block 4 MAE: 0.4039876010114124
- Error amplification ratio: 1.7483785560747438
