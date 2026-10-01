# Sampling3 pair-aware K–T factorial grid — preliminary one-seed results

## Setup

- `M_pairs=0.5` means one trajectory selected from one matched pair; `M_pairs>=1` means both trajectories from each selected pair.
- Pairing is variation `i` with variation `i+16`; the half-pair side is randomly selected and then held fixed across all T values within a draw.
- Grid: `M_pairs = 0.5, 1, 2, 4, 8`; `T = 32, 64, 128, 256`. Stored-point count is `2 × M_pairs × T` (so a half-pair uses `T` points).
- One training seed: `42`; max epochs `10`, patience `3`.
- Test predictions average 10 nested pair draws per system; intervals resample systems 2000 times.
- Bootstrap intervals quantify test-system uncertainty, not training-seed variability.

## Primary result: all-32 target, pair-DeepSets

### Fixed-T slices: MAE by observed phase allocation

| T | M=0.5 | M=1 | M=2 | M=4 | M=8 |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.325 [0.312, 0.337] | 0.283 [0.270, 0.297] | 0.264 [0.250, 0.278] | 0.252 [0.238, 0.266] | 0.246 [0.232, 0.260] |
| 64 | 0.314 [0.302, 0.326] | 0.238 [0.225, 0.252] | 0.218 [0.204, 0.232] | 0.207 [0.194, 0.221] | 0.201 [0.187, 0.215] |
| 128 | 0.304 [0.291, 0.315] | 0.199 [0.186, 0.211] | 0.180 [0.167, 0.194] | 0.174 [0.161, 0.187] | 0.168 [0.155, 0.181] |
| 256 | 0.293 [0.281, 0.306] | 0.166 [0.154, 0.178] | 0.152 [0.140, 0.166] | 0.148 [0.136, 0.160] | 0.146 [0.133, 0.159] |

### Fixed-M depth slices: MAE by prefix length

| M pairs | T=32 | T=64 | T=128 | T=256 |
|---:|---:|---:|---:|---:|
| 0.5 | 0.325 [0.312, 0.337] | 0.314 [0.302, 0.326] | 0.304 [0.291, 0.315] | 0.293 [0.281, 0.306] |
| 1 | 0.283 [0.270, 0.297] | 0.238 [0.225, 0.252] | 0.199 [0.186, 0.211] | 0.166 [0.154, 0.178] |
| 2 | 0.264 [0.250, 0.278] | 0.218 [0.204, 0.232] | 0.180 [0.167, 0.194] | 0.152 [0.140, 0.166] |
| 4 | 0.252 [0.238, 0.266] | 0.207 [0.194, 0.221] | 0.174 [0.161, 0.187] | 0.148 [0.136, 0.160] |
| 8 | 0.246 [0.232, 0.260] | 0.201 [0.187, 0.215] | 0.168 [0.155, 0.181] | 0.146 [0.133, 0.159] |

## How to read the contrasts

`pair_completion_at_fixed_T` compares one member (`M=0.5`) with both members of that same sampled pair (`M=1`). `breadth_at_fixed_T` compares `M=1` with more matched pairs (`M=2,4,8`). `depth_at_fixed_M` compares `T=32` with longer prefixes at the same M. Positive `delta_mae` favors the comparison allocation. `factorial_comparisons.csv` contains paired system-bootstrap intervals and Holm adjustments within each fixed-T or fixed-M slice.

This is a single-seed preliminary result. The grid separates phase allocation from prefix length, while the bootstrap does not address model-seed uncertainty.

## Files

- `config.json`, `pair_partition.csv`, `system_targets.csv`
- `per_system_predictions.csv`, `factorial_metrics.csv`, `factorial_comparisons.csv`
- `preinstability_eligibility.csv`
- `figure1_factorial_heatmap.png`, `figure2_factorial_slices.png`, `figure3_heldout_controls.png`
