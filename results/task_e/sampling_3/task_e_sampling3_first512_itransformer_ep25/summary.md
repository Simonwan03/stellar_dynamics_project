# Task E summary: `itransformer`

## E1 one-shot 512 -> 128

- MSE (z): 0.22687450470019488
- MAE (z): 0.2602822503630042
- RMSE (z): 0.47631345215120147
- Pearson: 0.8678868213876547
- Unit-circle error: 0.28850564008896373
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 0.22687450470019488, 'ci95_low': 0.21624382714125992, 'ci95_high': 0.23854252988742483, 'num_systems': 652}, 'mae_z': {'mean': 0.2602822503630042, 'ci95_low': 0.25173931384792453, 'ci95_high': 0.26963030790917925, 'num_systems': 652}, 'rmse_z': {'mean': 0.47631345215120147, 'ci95_low': 0.46502023819594673, 'ci95_high': 0.4884081588259206, 'num_systems': 652}}
- Sampling metadata: {'num_train_systems': 3037, 'num_val_systems': 650, 'num_test_systems': 652, 'num_train_trajectories': 62378, 'num_val_trajectories': 12713, 'num_test_trajectories': 13676, 'candidate_train_windows': 62375, 'candidate_val_windows': 12713, 'candidate_test_windows': 13674, 'candidate_train_windows_after_legacy_cap': 62375, 'candidate_val_windows_after_legacy_cap': 12713, 'candidate_test_windows_after_legacy_cap': 13674, 'sampled_train_windows_per_epoch': 62375, 'sampled_val_windows': 12713, 'sampled_e1_test_windows': 13674, 'planned_e2_rolling_windows': 12536, 'sampled_e2_rolling_windows': 12535, 'train_windows_per_trajectory': 1, 'val_windows_per_trajectory': 1, 'test_windows_per_trajectory': 1, 'rolling_windows_per_trajectory': 1, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system', 'last_train_epoch': 25, 'last_train_epoch_windows': 62375}

## E2 recursive rolling

- Windows: 12535
- Horizon: 512
- Block 1 MAE: 0.2551367121964531
- Block 2 MAE: 0.33985344991317
- Block 3 MAE: 0.3822618793589622
- Block 4 MAE: 0.4090719598563241
- Error amplification ratio: 1.6033441692285453
