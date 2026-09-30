# Sampling3 pair-aware K–T factorial grid — preliminary one-seed results

## Setup

- `M_pairs=0.5` means one trajectory selected from one matched pair; `M_pairs>=1` means both trajectories from each selected pair.
- Pairing is variation `i` with variation `i+16`; the half-pair side is randomly selected and then held fixed across all T values within a draw.
- Grid: `M_pairs = 0.5, 1, 2, 4, 8`; `T = 32, 64, 128, 256`. Stored-point count is `2 × M_pairs × T` (so a half-pair uses `T` points).
- One training seed: `42`; max epochs `1`, patience `1`.
- Test predictions average 2 nested pair draws per system; intervals resample systems 20 times.
- Bootstrap intervals quantify test-system uncertainty, not training-seed variability.

## Primary result: all-32 target, pair-DeepSets

### Fixed-T slices: MAE by observed phase allocation

| T | M=0.5 | M=1 | M=2 | M=4 | M=8 |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.352 [0.341, 0.360] | 0.303 [0.283, 0.309] | 0.289 [0.275, 0.300] | 0.283 [0.272, 0.298] | 0.277 [0.258, 0.288] |
| 64 | 0.340 [0.329, 0.351] | 0.269 [0.254, 0.278] | 0.250 [0.241, 0.269] | 0.240 [0.227, 0.255] | 0.232 [0.219, 0.249] |
| 128 | 0.331 [0.318, 0.339] | 0.231 [0.218, 0.241] | 0.212 [0.196, 0.232] | 0.203 [0.191, 0.213] | 0.198 [0.184, 0.212] |
| 256 | 0.324 [0.313, 0.338] | 0.208 [0.195, 0.219] | 0.190 [0.175, 0.197] | 0.184 [0.172, 0.202] | 0.179 [0.171, 0.194] |

### Fixed-M depth slices: MAE by prefix length

| M pairs | T=32 | T=64 | T=128 | T=256 |
|---:|---:|---:|---:|---:|
| 0.5 | 0.352 [0.341, 0.360] | 0.340 [0.329, 0.351] | 0.331 [0.318, 0.339] | 0.324 [0.313, 0.338] |
| 1 | 0.303 [0.283, 0.309] | 0.269 [0.254, 0.278] | 0.231 [0.218, 0.241] | 0.208 [0.195, 0.219] |
| 2 | 0.289 [0.275, 0.300] | 0.250 [0.241, 0.269] | 0.212 [0.196, 0.232] | 0.190 [0.175, 0.197] |
| 4 | 0.283 [0.272, 0.298] | 0.240 [0.227, 0.255] | 0.203 [0.191, 0.213] | 0.184 [0.172, 0.202] |
| 8 | 0.277 [0.258, 0.288] | 0.232 [0.219, 0.249] | 0.198 [0.184, 0.212] | 0.179 [0.171, 0.194] |

## How to read the contrasts

`pair_completion_at_fixed_T` compares one member (`M=0.5`) with both members of that same sampled pair (`M=1`). `breadth_at_fixed_T` compares `M=1` with more matched pairs (`M=2,4,8`). `depth_at_fixed_M` compares `T=32` with longer prefixes at the same M. Positive `delta_mae` favors the comparison allocation. `factorial_comparisons.csv` contains paired system-bootstrap intervals and Holm adjustments within each fixed-T or fixed-M slice.

This is a single-seed preliminary result. The grid separates phase allocation from prefix length, while the bootstrap does not address model-seed uncertainty.

## Files

- `config.json`, `pair_partition.csv`, `system_targets.csv`
- `per_system_predictions.csv`, `factorial_metrics.csv`, `factorial_comparisons.csv`
- `preinstability_eligibility.csv`
- `figure1_factorial_heatmap.png`, `figure2_factorial_slices.png`, `figure3_heldout_controls.png`
