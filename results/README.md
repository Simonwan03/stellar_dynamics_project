# Experiment artifacts

All saved experiment outputs live here, grouped by paper task and then source
dataset. The old `runs/` tree has been merged into this directory.

| Folder | Contents |
| --- | --- |
| `task_a/` | System-level stability and stable-fraction prediction |
| `task_b/` | Short-prefix stability prediction |
| `task_c/` | Survival-curve prediction |
| `task_d/` | Finite-time divergence and historical proxy experiments |
| `task_e/` | Trajectory forecasting, including prepared caches and model runs |
| `task_f/` | Partial-observation robustness; `sampling_3/masked_forecasting/` contains masked Task E-style forecasting runs used for Task F |
| `comparisons/` | Cross-task A/B/C comparisons and observation-budget studies |
| `shared_analysis/` | Dataset audits and figures shared across tasks |

`sampling_2` refers to `dataset_v3_low_resolution`; `sampling_3` refers to
`dataset_v3_high_resolution`. `dataset_unknown` means the saved metadata do
not establish the source dataset. Use the [artifact inventory](../docs/artifact_inventory.md)
for individual runs and the [manuscript reproduction map](../docs/reproduction.md)
for the final paper models. The [merge map](../docs/artifact_merge_map.csv)
records paths from before the `results/`/`runs/` consolidation.

Generated files are ignored by Git unless explicitly added. Existing tracked
configs, metrics, and figures were moved to keep their provenance available.
Saved text configs now point to the new locations. Binary checkpoints were
left byte-for-byte intact; their embedded historical config may still mention
`runs/`, while the current CLI uses the checkpoint path supplied at runtime.
