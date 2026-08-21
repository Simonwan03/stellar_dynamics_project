# Task E summary: `tirex2`

## E1 one-shot 512 -> 128

- MSE (z): 0.24216084731371817
- MAE (z): 0.2407973287319937
- RMSE (z): 0.49209841222434175
- Pearson: 0.8508211149572729
- Unit-circle error: 0.2277437993210985
- Primary aggregation: equal-weight system -> trajectory -> window.
- System bootstrap 95% CI: {'mse_z': {'mean': 0.24216084731371817, 'ci95_low': 0.22896888542730112, 'ci95_high': 0.2555025431468406, 'num_systems': 578}, 'mae_z': {'mean': 0.2407973287319937, 'ci95_low': 0.22972122270028117, 'ci95_high': 0.25234492055909724, 'num_systems': 578}, 'rmse_z': {'mean': 0.49209841222434175, 'ci95_low': 0.47850693349096096, 'ci95_high': 0.5054725937790007, 'num_systems': 578}}
- Sampling metadata: {'num_train_systems': 2692, 'num_val_systems': 577, 'num_test_systems': 578, 'num_train_trajectories': 44988, 'num_val_trajectories': 9961, 'num_test_trajectories': 9648, 'candidate_train_windows': 2678116, 'candidate_val_windows': 600998, 'candidate_test_windows': 579577, 'candidate_train_windows_after_legacy_cap': 2678116, 'candidate_val_windows_after_legacy_cap': 600998, 'candidate_test_windows_after_legacy_cap': 579577, 'sampled_train_windows_per_epoch': 351944, 'sampled_val_windows': 39518, 'sampled_e1_test_windows': 38245, 'planned_e2_rolling_windows': 17488, 'sampled_e2_rolling_windows': 17485, 'train_windows_per_trajectory': 8, 'val_windows_per_trajectory': 4, 'test_windows_per_trajectory': 4, 'rolling_windows_per_trajectory': 2, 'train_sampling': 'dynamic_random_per_trajectory', 'validation_sampling': 'deterministic_positions_10_35_65_90', 'e1_test_sampling': 'deterministic_positions_10_35_65_90', 'e2_rolling_sampling': 'deterministic_positions_25_75', 'aggregation_unit': 'system', 'hierarchy': 'system -> trajectory -> forecasting_window', 'bootstrap_unit': 'system'}

## E2 recursive rolling

- Windows: 17485
- Horizon: 512
- Block 1 MAE: 0.23523958621268684
- Block 2 MAE: 0.3039419914508007
- Block 3 MAE: 0.3310460662314893
- Block 4 MAE: 0.3507110825598675
- Error amplification ratio: 1.490867622266515
