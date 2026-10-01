# Artifact inventory

All experiment outputs are under `results/`, grouped by task and source dataset. Task E forecasting runs are in `results/task_e/`; Task F masked forecasting runs are in `results/task_f/sampling_3/masked_forecasting/`. Cross-task comparisons are in `results/comparisons/`, and shared figures and audits are in `results/shared_analysis/`. `dataset_unknown` marks artifacts without a verified source dataset. The [path map](artifact_path_map.csv) records original and current locations; the [merge map](artifact_merge_map.csv) records the `results/`/`runs/` consolidation. Artifact type is inferred from names and saved files; see the manuscript reproduction map for final model selection.

Regenerate with: python tools/build_artifact_inventory.py. The script reads saved JSON metadata and directory names. It does not open HDF5 shards or rewrite results. Here dataset_v3 means sampling_2.

| Task | Dataset | Items |
| --- | --- | ---: |
| A | sampling_2 | 1 |
| A+B | sampling_3 | 5 |
| A+B | unknown | 1 |
| A+B+C | unknown | 1 |
| B | sampling_2 | 1 |
| B | sampling_3 | 17 |
| C | sampling_2 | 3 |
| C | sampling_3 | 4 |
| D | sampling_2 | 6 |
| D | sampling_3 | 8 |
| E | sampling_2 | 24 |
| E | sampling_3 | 11 |
| E | unknown | 2 |
| F | sampling_3 | 25 |
| shared/analysis | sampling_2 | 3 |
| shared/analysis | sampling_2+sampling_3 | 1 |
| shared/analysis | sampling_3 | 1 |
| shared/analysis | unknown | 1 |

## A / sampling_2

- [results/task_a/sampling_2/task_a_common](../results/task_a/sampling_2/task_a_common) - other artifact

## A+B / sampling_3

- [results/comparisons/sampling_3/factorial_sampling3_Kpairs0p5-8_T32-256](../results/comparisons/sampling_3/factorial_sampling3_Kpairs0p5-8_T32-256) - metrics/run
- [results/comparisons/sampling_3/factorial_sampling3_halfpair_smoke](../results/comparisons/sampling_3/factorial_sampling3_halfpair_smoke) - exploratory
- [results/comparisons/sampling_3/fixed_budget_sampling3_B512](../results/comparisons/sampling_3/fixed_budget_sampling3_B512) - metrics/run
- [results/comparisons/sampling_3/fixed_budget_sampling3_B512_pair_aware](../results/comparisons/sampling_3/fixed_budget_sampling3_B512_pair_aware) - metrics/run
- [results/comparisons/sampling_3/fixed_budget_sampling3_B512_pair_aware_smoke](../results/comparisons/sampling_3/fixed_budget_sampling3_B512_pair_aware_smoke) - exploratory

## A+B / unknown

- [results/comparisons/dataset_unknown/task_ab_future_only_ablation](../results/comparisons/dataset_unknown/task_ab_future_only_ablation) - other artifact; source: unverified

## A+B+C / unknown

- [results/comparisons/dataset_unknown/abc_common_target_h10000_prefix96](../results/comparisons/dataset_unknown/abc_common_target_h10000_prefix96) - other artifact; source: unverified

## B / sampling_2

- [results/task_b/sampling_2/task_b_common](../results/task_b/sampling_2/task_b_common) - other artifact

## B / sampling_3

- [results/task_b/sampling_3/task_b_sampling3_prefix128](../results/task_b/sampling_3/task_b_sampling3_prefix128) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix128_chronos2](../results/task_b/sampling_3/task_b_sampling3_prefix128_chronos2) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix128_panda_linear](../results/task_b/sampling_3/task_b_sampling3_prefix128_panda_linear) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix128_stats_xgboost](../results/task_b/sampling_3/task_b_sampling3_prefix128_stats_xgboost) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix256](../results/task_b/sampling_3/task_b_sampling3_prefix256) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix256_chronos2](../results/task_b/sampling_3/task_b_sampling3_prefix256_chronos2) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix256_panda_linear](../results/task_b/sampling_3/task_b_sampling3_prefix256_panda_linear) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix256_stats_xgboost](../results/task_b/sampling_3/task_b_sampling3_prefix256_stats_xgboost) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix512](../results/task_b/sampling_3/task_b_sampling3_prefix512) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix512_chronos2](../results/task_b/sampling_3/task_b_sampling3_prefix512_chronos2) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix512_panda_linear](../results/task_b/sampling_3/task_b_sampling3_prefix512_panda_linear) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix512_stats_xgboost](../results/task_b/sampling_3/task_b_sampling3_prefix512_stats_xgboost) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix64](../results/task_b/sampling_3/task_b_sampling3_prefix64) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix64_chronos2](../results/task_b/sampling_3/task_b_sampling3_prefix64_chronos2) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix64_panda_linear](../results/task_b/sampling_3/task_b_sampling3_prefix64_panda_linear) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_prefix64_stats_xgboost](../results/task_b/sampling_3/task_b_sampling3_prefix64_stats_xgboost) - metrics/run
- [results/task_b/sampling_3/task_b_sampling3_two_level_evaluation](../results/task_b/sampling_3/task_b_sampling3_two_level_evaluation) - other artifact; source: directory name

## C / sampling_2

- [results/task_c/sampling_2/task_c_survival20_common_prefix96_chronos](../results/task_c/sampling_2/task_c_survival20_common_prefix96_chronos) - metrics/run
- [results/task_c/sampling_2/task_c_survival20_common_prefix96_panda](../results/task_c/sampling_2/task_c_survival20_common_prefix96_panda) - metrics/run
- [results/task_c/sampling_2/task_c_survival20_common_prefix96_raw](../results/task_c/sampling_2/task_c_survival20_common_prefix96_raw) - metrics/run

## C / sampling_3

- [results/task_c/sampling_3/task_c_sampling3_prefix128_cache](../results/task_c/sampling_3/task_c_sampling3_prefix128_cache) - cache/checkpoint
- [results/task_c/sampling_3/task_c_sampling3_prefix128_chronos2](../results/task_c/sampling_3/task_c_sampling3_prefix128_chronos2) - metrics/run
- [results/task_c/sampling_3/task_c_sampling3_prefix128_panda](../results/task_c/sampling_3/task_c_sampling3_prefix128_panda) - metrics/run
- [results/task_c/sampling_3/task_c_sampling3_prefix128_raw](../results/task_c/sampling_3/task_c_sampling3_prefix128_raw) - metrics/run

## D / sampling_2

- [results/task_d/sampling_2/divergence_statistics](../results/task_d/sampling_2/divergence_statistics) - analysis
- [results/task_d/sampling_2/task_d_initial_lambda_fast](../results/task_d/sampling_2/task_d_initial_lambda_fast) - metrics/run
- [results/task_d/sampling_2/task_d_initial_lambda_fast_orbital15](../results/task_d/sampling_2/task_d_initial_lambda_fast_orbital15) - metrics/run
- [results/task_d/sampling_2/task_d_initial_lyapunov_proxy](../results/task_d/sampling_2/task_d_initial_lyapunov_proxy) - metrics/run
- [results/task_d/sampling_2/task_d_initial_prefix_lambda_fast_orbital15](../results/task_d/sampling_2/task_d_initial_prefix_lambda_fast_orbital15) - metrics/run
- [results/task_d/sampling_2/task_d_proxy_sensitivity](../results/task_d/sampling_2/task_d_proxy_sensitivity) - other artifact

## D / sampling_3

- [results/task_d/sampling_3/task_d_first_variation_mlp_xgboost](../results/task_d/sampling_3/task_d_first_variation_mlp_xgboost) - metrics/run
- [results/task_d/sampling_3/task_d_initial_quantile_sensitivity](../results/task_d/sampling_3/task_d_initial_quantile_sensitivity) - other artifact
- [results/task_d/sampling_3/task_d_initial_quantile_sensitivity_mlp_xgboost](../results/task_d/sampling_3/task_d_initial_quantile_sensitivity_mlp_xgboost) - other artifact
- [results/task_d/sampling_3/task_d_initial_quantiles](../results/task_d/sampling_3/task_d_initial_quantiles) - metrics/run
- [results/task_d/sampling_3/task_d_median_decorrelation](../results/task_d/sampling_3/task_d_median_decorrelation) - metrics/run
- [results/task_d/sampling_3/task_d_sampling3_pair_efold_float64](../results/task_d/sampling_3/task_d_sampling3_pair_efold_float64) - other artifact
- [results/task_d/sampling_3/task_d_sampling3_q_epowers](../results/task_d/sampling_3/task_d_sampling3_q_epowers) - other artifact
- [results/task_d/sampling_3/task_d_sampling3_q_multithreshold](../results/task_d/sampling_3/task_d_sampling3_q_multithreshold) - other artifact

## E / sampling_2

- [results/task_e/sampling_2/panda_temporal_adapter_full](../results/task_e/sampling_2/panda_temporal_adapter_full) - other artifact
- [results/task_e/sampling_2/panda_temporal_adapter_full_20k](../results/task_e/sampling_2/panda_temporal_adapter_full_20k) - metrics/run
- [results/task_e/sampling_2/task_e_chronos2_orbital15_ctx512_pred128](../results/task_e/sampling_2/task_e_chronos2_orbital15_ctx512_pred128) - metrics/run
- [results/task_e/sampling_2/task_e_e1e2_summary](../results/task_e/sampling_2/task_e_e1e2_summary) - other artifact; source: linked run configuration
- [results/task_e/sampling_2/task_e_itransformer_orbital15_ctx512_pred128](../results/task_e/sampling_2/task_e_itransformer_orbital15_ctx512_pred128) - metrics/run
- [results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_chronos_t5_small_2var2win](../results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_chronos_t5_small_2var2win) - metrics/run
- [results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_gru](../results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_gru) - other artifact
- [results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_gru_2var2win](../results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_gru_2var2win) - metrics/run
- [results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_gru_2var2win_earlystop](../results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_gru_2var2win_earlystop) - metrics/run
- [results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_mlp_2var2win](../results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_mlp_2var2win) - other artifact
- [results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_mlp_2var2win_epoch15](../results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_mlp_2var2win_epoch15) - metrics/run
- [results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_zero_shot_2var2win](../results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_panda_zero_shot_2var2win) - metrics/run
- [results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_timesfm_2var2win](../results/task_e/sampling_2/task_e_orbital15_ctx64_pred32_timesfm_2var2win) - metrics/run
- [results/task_e/sampling_2/task_e_panda_zero_shot_orbital15_ctx512_pred128](../results/task_e/sampling_2/task_e_panda_zero_shot_orbital15_ctx512_pred128) - metrics/run
- [results/task_e/sampling_2/task_e_prelim_compact_20260712](../results/task_e/sampling_2/task_e_prelim_compact_20260712) - exploratory
- [results/task_e/sampling_2/task_e_prelim_compact_20260712_v2](../results/task_e/sampling_2/task_e_prelim_compact_20260712_v2) - exploratory
- [results/task_e/sampling_2/task_e_prelim_small_20260712](../results/task_e/sampling_2/task_e_prelim_small_20260712) - exploratory
- [results/task_e/sampling_2/task_e_prelim_small_20260712_seq](../results/task_e/sampling_2/task_e_prelim_small_20260712_seq) - exploratory
- [results/task_e/sampling_2/task_e_prelim_small_20260712_v2](../results/task_e/sampling_2/task_e_prelim_small_20260712_v2) - exploratory
- [results/task_e/sampling_2/task_e_prepared_cache_orbital15_ctx512_pred128](../results/task_e/sampling_2/task_e_prepared_cache_orbital15_ctx512_pred128) - cache/checkpoint
- [results/task_e/sampling_2/task_e_raw18_matched_20260713](../results/task_e/sampling_2/task_e_raw18_matched_20260713) - other artifact
- [results/task_e/sampling_2/task_e_split_checkpoint_orbital15_ctx512_pred128](../results/task_e/sampling_2/task_e_split_checkpoint_orbital15_ctx512_pred128) - cache/checkpoint
- [results/task_e/sampling_2/task_e_tirex2_orbital15_ctx512_pred128](../results/task_e/sampling_2/task_e_tirex2_orbital15_ctx512_pred128) - metrics/run
- [results/task_e/sampling_2/task_e_toto2_orbital15_ctx512_pred128](../results/task_e/sampling_2/task_e_toto2_orbital15_ctx512_pred128) - metrics/run

## E / sampling_3

- [results/task_e/sampling_3/task_e_sampling3_first512_chronos2](../results/task_e/sampling_3/task_e_sampling3_first512_chronos2) - metrics/run
- [results/task_e/sampling_3/task_e_sampling3_first512_itransformer](../results/task_e/sampling_3/task_e_sampling3_first512_itransformer) - other artifact
- [results/task_e/sampling_3/task_e_sampling3_first512_itransformer_ep20](../results/task_e/sampling_3/task_e_sampling3_first512_itransformer_ep20) - other artifact
- [results/task_e/sampling_3/task_e_sampling3_first512_itransformer_ep25](../results/task_e/sampling_3/task_e_sampling3_first512_itransformer_ep25) - metrics/run
- [results/task_e/sampling_3/task_e_sampling3_first512_linear](../results/task_e/sampling_3/task_e_sampling3_first512_linear) - metrics/run
- [results/task_e/sampling_3/task_e_sampling3_first512_panda](../results/task_e/sampling_3/task_e_sampling3_first512_panda) - metrics/run
- [results/task_e/sampling_3/task_e_sampling3_first512_prepared](../results/task_e/sampling_3/task_e_sampling3_first512_prepared) - cache/checkpoint
- [results/task_e/sampling_3/task_e_sampling3_first512_tcn_ep25](../results/task_e/sampling_3/task_e_sampling3_first512_tcn_ep25) - other artifact
- [results/task_e/sampling_3/task_e_sampling3_first512_tcn_ep25_continued](../results/task_e/sampling_3/task_e_sampling3_first512_tcn_ep25_continued) - metrics/run
- [results/task_e/sampling_3/task_e_sampling3_first512_tirex2](../results/task_e/sampling_3/task_e_sampling3_first512_tirex2) - metrics/run
- [results/task_e/sampling_3/task_e_sampling3_first512_toto2](../results/task_e/sampling_3/task_e_sampling3_first512_toto2) - metrics/run

## E / unknown

- [results/task_e/dataset_unknown/task_e_e1e2_tcn_orbital15_ctx512_pred128](../results/task_e/dataset_unknown/task_e_e1e2_tcn_orbital15_ctx512_pred128) - other artifact; source: unverified
- [results/task_e/dataset_unknown/task_e_orbital15_ctx64_pred32_standardized_horizon_comparison.png](../results/task_e/dataset_unknown/task_e_orbital15_ctx64_pred32_standardized_horizon_comparison.png) - standalone file; source: unverified

## F / sampling_3

- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_chronos2_masked_r25](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_chronos2_masked_r25) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_chronos2_masked_r50](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_chronos2_masked_r50) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_chronos2_masked_r75](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_chronos2_masked_r75) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_itransformer_masked_clean_r50](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_itransformer_masked_clean_r50) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_itransformer_masked_mixed_eval_r25](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_itransformer_masked_mixed_eval_r25) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_itransformer_masked_mixed_eval_r75](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_itransformer_masked_mixed_eval_r75) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_itransformer_masked_mixed_r50](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_itransformer_masked_mixed_r50) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_panda_masked_r25](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_panda_masked_r25) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_panda_masked_r50](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_panda_masked_r50) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_panda_masked_r75](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_panda_masked_r75) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tcn_masked_clean_r50](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tcn_masked_clean_r50) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tcn_masked_mixed_eval_r25](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tcn_masked_mixed_eval_r25) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tcn_masked_mixed_eval_r75](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tcn_masked_mixed_eval_r75) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tcn_masked_mixed_r50](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tcn_masked_mixed_r50) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tirex2_masked_r25](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tirex2_masked_r25) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tirex2_masked_r50](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tirex2_masked_r50) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tirex2_masked_r75](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_tirex2_masked_r75) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_toto2_masked_r25](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_toto2_masked_r25) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_toto2_masked_r50](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_toto2_masked_r50) - metrics/run
- [results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_toto2_masked_r75](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_toto2_masked_r75) - metrics/run
- [results/task_f/sampling_3/task_f_contiguous1](../results/task_f/sampling_3/task_f_contiguous1) - metrics/run
- [results/task_f/sampling_3/task_f_masked_prefix](../results/task_f/sampling_3/task_f_masked_prefix) - metrics/run
- [results/task_f/sampling_3/task_f_masked_prefix_attention](../results/task_f/sampling_3/task_f_masked_prefix_attention) - metrics/run
- [results/task_f/sampling_3/task_f_mixed_mask_benchmark_seed42](../results/task_f/sampling_3/task_f_mixed_mask_benchmark_seed42) - metrics/run
- [results/task_f/sampling_3/task_f_summary_xgboost](../results/task_f/sampling_3/task_f_summary_xgboost) - metrics/run

## shared/analysis / sampling_2

- [results/shared_analysis/sampling_2/dataset_v3_orbital_examples](../results/shared_analysis/sampling_2/dataset_v3_orbital_examples) - exploratory; source: directory name
- [results/shared_analysis/sampling_2/system_level_audit_sampling_2](../results/shared_analysis/sampling_2/system_level_audit_sampling_2) - analysis
- [results/shared_analysis/sampling_2/traditional_stability](../results/shared_analysis/sampling_2/traditional_stability) - analysis

## shared/analysis / sampling_2+sampling_3

- [results/shared_analysis/sampling_2+sampling_3/time_unbound_distribution_sampling_2_3](../results/shared_analysis/sampling_2+sampling_3/time_unbound_distribution_sampling_2_3) - analysis; source: directory name and summary

## shared/analysis / sampling_3

- [results/shared_analysis/sampling_3/system_level_audit_sampling_3](../results/shared_analysis/sampling_3/system_level_audit_sampling_3) - analysis

## shared/analysis / unknown

- [results/shared_analysis/dataset_unknown/analysis](../results/shared_analysis/dataset_unknown/analysis) - analysis; source: unverified
