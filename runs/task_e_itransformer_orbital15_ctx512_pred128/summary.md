# Task E summary: `itransformer`

## E1 one-shot 512 -> 128

- MSE (z): 0.21234861347284872
- MAE (z): 0.24293063950554197
- RMSE (z): 0.4608129918663847
- Pearson: 0.8672591433712716
- Unit-circle error: 0.31518700621684337
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 0.21234861347284872, 'ci95_low': 0.2016633566020231, 'ci95_high': 0.22299718727700735, 'num_systems': 578}, 'mae_z': {'mean': 0.24293063950554197, 'ci95_low': 0.23259772014095664, 'ci95_high': 0.25374828851331055, 'num_systems': 578}, 'rmse_z': {'mean': 0.4608129918663847, 'ci95_low': 0.44906942857355536, 'ci95_high': 0.47222577956619144, 'num_systems': 578}}
- Sampling metadata: {'num_train_systems': 2692, 'num_val_systems': 577, 'num_test_systems': 578, 'num_train_trajectories': 44988, 'num_val_trajectories': 9961, 'num_test_trajectories': 9648, 'candidate_train_windows': 2678116, 'candidate_val_windows': 600998, 'candidate_test_windows': 579577, 'candidate_train_windows_after_legacy_cap': 2678116, 'candidate_val_windows_after_legacy_cap': 600998, 'candidate_test_windows_after_legacy_cap': 579577, 'sampled_train_windows_per_epoch': 351944, 'sampled_val_windows': 39518, 'sampled_e1_test_windows': 38245, 'planned_e2_rolling_windows': 17488, 'sampled_e2_rolling_windows': 17485, 'train_windows_per_trajectory': 8, 'val_windows_per_trajectory': 4, 'test_windows_per_trajectory': 4, 'rolling_windows_per_trajectory': 2, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system', 'last_train_epoch': 30, 'last_train_epoch_windows': 351944}

## E2 recursive rolling

- Windows: 17485
- Horizon: 512
- Block 1 MAE: 0.23796642631170323
- Block 2 MAE: 0.29930594749674494
- Block 3 MAE: 0.3294189012676691
- Block 4 MAE: 0.3515242481900597
- Error amplification ratio: 1.4772010221711338
