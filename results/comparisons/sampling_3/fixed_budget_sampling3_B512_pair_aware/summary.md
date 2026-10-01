# Sampling3 fixed-budget matched-pair experiment — preliminary one-seed result

## Design

- Pair mapping: variation `i` is matched with variation `i+16`, for `i=0,...,15`.
- A sampled unit is one whole pair. The pair-aware regressor encodes both members, forms a symmetric pair representation (member mean and absolute difference), then pools across observed pairs.
- Fixed stored-point budget: `2 × M_pairs × T = 512`, giving `(M_pairs, trajectories, T) = (1,2,256), (2,4,128), (4,8,64), (8,16,32)`.
- Held-out control splits each system's 16 pairs into 8 candidate and 8 held-out pairs; both members of each pair stay on the same side.
- One training seed (`42`), 50 nested pair draws per test system, and 2,000 paired system bootstrap replicates. To get a quick result, the regressor was trained on the four fixed-budget allocations only, with at most 10 epochs and patience 3. This is a preliminary run, not the planned five-seed result.

## Test MAE

Numbers in brackets are 95% system-bootstrap intervals. Lower is better.

| Observed matched pairs | Trajectories | Points/trajectory | DeepSets, all-32 target | DeepSets, held-out target | TCN mean-probability, all-32 target | DeepSets, held-out + pre-instability |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 256 | 0.168 [0.157, 0.180] | 0.164 [0.152, 0.176] | 0.266 [0.252, 0.280] | 0.235 [0.219, 0.252] |
| 2 | 4 | 128 | 0.180 [0.167, 0.193] | 0.174 [0.162, 0.187] | 0.275 [0.261, 0.289] | 0.260 [0.243, 0.277] |
| 4 | 8 | 64 | 0.211 [0.197, 0.224] | 0.205 [0.191, 0.221] | 0.282 [0.268, 0.298] | 0.300 [0.281, 0.319] |
| 8 | 16 | 32 | 0.256 [0.242, 0.271] | 0.250 [0.234, 0.266] | 0.288 [0.271, 0.303] | 0.344 [0.326, 0.362] |

## Initial interpretation

For this seed, increasing the number of observed pairs did not improve the estimate. The best result is one pair observed to 256 points per trajectory. For the all-32 target, the DeepSets paired difference `MAE(M=1) - MAE(M)` is `-0.0115` for `M=2` (95% CI `[-0.0180,-0.0054]`), `-0.0426` for `M=4` (`[-0.0533,-0.0325]`), and `-0.0878` for `M=8` (`[-0.1019,-0.0737]`); all three Holm-adjusted bootstrap p-values are `0.0015`. The held-out target and held-out + pre-instability control show the same direction. The simple probability-averaging baseline also worsens as more pairs are allocated shorter prefixes.

The pre-instability control contains 480 test systems with at least one eligible draw; 20,269 of 37,500 candidate-pair system-draws were eligible across all four budget conditions. Thus the observed decline is not explained solely by already-failed observed trajectories in this control.

This is an early signal favoring temporal depth under this budget and training setup, not yet a final scientific conclusion. It uses one model seed; the bootstrap does not measure training-seed variability. The held-out DeepSets training also reached the 10-epoch cap, so a multi-seed run with longer training is needed before interpreting the ranking as robust. MA-boundary subgroup results are in `metrics.csv` and `paired_bootstrap_comparisons.csv`; treat them as exploratory at this stage.

## Artifacts

- `config.json`: run settings and seeds
- `pair_partition.csv`: per-system candidate/held-out pair assignments and variation IDs
- `per_system_predictions.csv`, `metrics.csv`, `paired_bootstrap_comparisons.csv`
- `preinstability_eligibility.csv`
- `figure1_fixed_budget_tradeoff.png`, `figure2_hard_controls.png`, `figure3_ma_boundary_breadth_gain.png`

The earlier `results/comparisons/sampling_3/fixed_budget_sampling3_B512/` run sampled individual variations and did not preserve the matched-pair structure. Use this pair-aware directory for the current preliminary result; the earlier run is not a pair-aware replication.
