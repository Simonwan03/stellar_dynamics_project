# Manuscript reproduction map

Run commands from the repository root. Dataset names map to the existing code paths as follows: sampling 2 = `dataset_v3_low_resolution` → `dataset_v3/`; sampling 3 = `dataset_v3_high_resolution` → `dataset_sampling_3_prefix1024/`. The sampling 3 data directory must contain the high-resolution HDF5 shards and matching completion table. Those files are untracked; this development machine links `dataset_sampling_3_prefix1024` to `/data/dataset_sampling_3_prefix1024_float64`. The `dataset_v3` symlink targets a developer machine, so supply sampling 2 data separately. Saved result `config.json` files record exact reported invocations; current CLI defaults can differ. Training entry points also evaluate validation and test unless an evaluator is named below.

| Task | Current pipeline | Reported settings and result provenance |
| --- | --- | --- |
| A: system stable fraction | `scripts/task_a.py --resolution low` loads 32 initial-condition variations per system. `tasks/common/case_split.py` assigns complete `raw_case_id` systems. | Sampling 2 MLP, XGBoost and analytic baselines; `--target-mode stable_fraction`. See [Task A research log](../record.md). `binary_any` is a different target. No sampling 3 Task A result config or required IC file is present. |
| B: short-prefix stability | `scripts/task_b.py --resolution high` loads sampling 3 orbital15 prefixes; `--resolution low` selects the sampling 2 Task B implementation. `experiments/foundation_models/shared/train_spock_inspired_task_b_xgboost.py` builds summary features; Panda and Chronos-2 encoder/head drivers are under `experiments/foundation_models/panda/` and `experiments/foundation_models/chronos2/`. | Prefixes 64/128/256/512; horizon 10000; [prefix 128 neural config](../results/task_b/sampling_3/task_b_sampling3_prefix128/config.json) and [Summary+XGBoost config](../results/task_b/sampling_3/task_b_sampling3_prefix128_stats_xgboost/run_config.json). `tasks/task_b/evaluate_highres_two_level.py` computes system stable-fraction metrics from trajectory predictions. Frozen embeddings and weights are external. |
| C: survival curve | `scripts/task_c.py --resolution high` constructs censored discrete hazards from complete systems and aligned Task B prefix data; `--resolution low` selects sampling 2 data with the original CLI defaults. | Sampling 3 prefix 128, 20 bins from 1 to 10000, seven readout horizons, 3498/750/750 system split. [Raw](../results/task_c/sampling_3/task_c_sampling3_prefix128_raw/config.json), [Panda](../results/task_c/sampling_3/task_c_sampling3_prefix128_panda/config.json), [Chronos-2](../results/task_c/sampling_3/task_c_sampling3_prefix128_chronos2/config.json). `tasks.task_c` implements the older fraction objective. |
| D: finite-time divergence | `scripts/task_d.py --resolution high` calls `tasks/task_d/quantile.py`. It derives paired semi-major-axis crossing times and one initial-condition vector per system. | Default exp(5) threshold, eight valid pairs, interpolated system median target (the current entry does not fit multiple quantiles), 70/15/15 split; constant, XGBoost and MLP. Evaluation, bootstrap and split CSV are written during training. Sampling 2 `tasks/task_d/pipeline.py` is a different historical target and is not exposed through `--resolution low`. |
| E: trajectory forecasting | `scripts/task_e.py --resolution low/high` selects the data files for one run; `tasks.task_e.prepare` builds a completion-filtered, train-only scaled, one-start cache. `bash scripts/task_e.sh <model>` is the full sampling 3 E1/E2 paper sequence. | Sampling 3 first 512 to next 128, orbital15, system-balanced metrics. Saved configs: [Chronos-2](../results/task_e/sampling_3/task_e_sampling3_first512_chronos2/config.json), [Panda](../results/task_e/sampling_3/task_e_sampling3_first512_panda/config.json), [Toto-2](../results/task_e/sampling_3/task_e_sampling3_first512_toto2/config.json), [iTransformer](../results/task_e/sampling_3/task_e_sampling3_first512_itransformer_ep25/config.json), [TCN](../results/task_e/sampling_3/task_e_sampling3_first512_tcn_ep25_continued/config.json). |
| F: partial observation | `scripts/task_f.py --resolution high` trains masked-prefix TCN/attention with the committed system split; `tasks/task_f/summary_xgboost.py` evaluates summary XGBoost; `tasks/task_f/mixed_mask_benchmark.py` compares mask regimes. Task E `--mask-regime` covers masked forecasting. | Sampling 3 prefix 256, orbital15 plus observed flag, horizon 10000. [Random mask](../results/task_f/sampling_3/task_f_masked_prefix/config.json), [mixed mask](../results/task_f/sampling_3/task_f_mixed_mask_benchmark_seed42/config.json), [masked iTransformer forecasting](../results/task_f/sampling_3/masked_forecasting/task_e_sampling3_first512_itransformer_masked_mixed_r50/config.json). Keep clean, random, block and mixed protocols separate. |

## Task B/C prefix input

The reported sampling-3 Task B and C prefix experiments use `orbital15` at each observed time step: `aratios_complete` (1), `mratios_complete` (2), `eccs_complete` (2), `incs_complete` (2), `sin/cos(APs_complete)` (4), and `sin/cos(LANs_complete)` (4). This totals 15 channels. `incs_complete` is kept in its original form. No pairwise geometry or normalized-time channel is appended. Task B's high-resolution runner uses this representation directly; Task C's raw heads use it directly, while the Panda and Chronos-2 heads consume frozen embeddings generated from the same prefixes.

The older wider input remains available via `python scripts/task_b.py --resolution low --representation full` and `python -m tasks.task_c.survival --prefix-representation full`. The optional geometry and time features belong to that `full` path. See [the research log's representation section](../record.md#2-prefix-orbital15-representation) for the field breakdown.

## Before running an experiment

Run `python -m tasks.common.preflight --task B --dataset-dir /path/to/sampling3` (substitute the desired task A–F). It checks the expected local inputs and, for B/C/F, validates the [committed sampling-3 system split](../configs/splits/README.md) against the completion table. It does not check pretrained model weights or train a model. Exact sampling-3 reproduction expects 40 shards with 1,024 time steps and the recorded completion table; see the [release audit](reproducibility_audit.md) for unresolved requirements.

## Training and evaluation commands

The following are verified entry points, not claims that the external data or weights are available. Consult each linked config to reproduce a particular model/table row exactly; pass its options to the matching entry point. These commands write evaluation metrics during training.

```bash
# A: sampling 2 stable fraction
python scripts/task_a.py --resolution low --dataset-dir dataset_v3 --target-mode stable_fraction --models mlp,xgboost --output-dir results/task_a/sampling_2/reproduce_task_a

# B: sampling 3 prefix 128 MLP/CNN/TCN
python scripts/task_b.py --resolution high --dataset-dir dataset_sampling_3_prefix1024 --prefix-len 128 --models mlp,cnn,tcn --split-manifest configs/splits/common_case_split_sampling_3_g32_h10000_r70-15-15_seed42.csv --output-dir results/task_b/sampling_3/reproduce_task_b

# C: sampling 3 raw/summary survival heads
python scripts/task_c.py --resolution high --dataset-dir dataset_sampling_3_prefix1024 --prefix-len 128 --hazard-t-min 1 --hazard-t-max 10000 --hazard-bins 20 --variation-target full32 --split-manifest configs/splits/common_case_split_sampling_3_g32_h10000_r70-15-15_seed42.csv --epochs 20 --models baseline,stats,cnn,tcn,gru,attention --output-dir results/task_c/sampling_3/reproduce_task_c

# D: median divergence-time regression
python scripts/task_d.py --resolution high --dataset-dir dataset_sampling_3_prefix1024 --models constant,xgboost,mlp --output-dir results/task_d/sampling_3/reproduce_task_d

# E: prepare one-start cache, then train/evaluate E1 and E2
bash scripts/task_e.sh chronos2 dataset_sampling_3_prefix1024

# F: random-mask prefix classification
python scripts/task_f.py --resolution high --dataset-dir dataset_sampling_3_prefix1024 --prefix-len 256 --mask-ratios 0,25,50,75 --models tcn,attention --split-manifest configs/splits/common_case_split_sampling_3_g32_h10000_r70-15-15_seed42.csv --output-dir results/task_f/sampling_3/reproduce_task_f
```

For the reported sampling 3 Summary+XGBoost row, use `python experiments/foundation_models/shared/train_spock_inspired_task_b_xgboost.py --dataset-dir dataset_sampling_3_prefix1024 --sampling-id 3 --prefix-dir results/task_b/sampling_3/task_b_sampling3_prefix512 --prefix-steps 128 --output-dir results/task_b/sampling_3/reproduce_task_b_stats_xgboost`. Task B's separate system-level evaluator reads the run's prediction files; use `python -m tasks.task_b.evaluate_highres_two_level --help` to select inputs and outputs. Frozen Panda and Chronos-2 B/C rows need aligned embeddings prepared by scripts in `experiments/foundation_models/panda/` and `experiments/foundation_models/chronos2/`. Cache metadata records label fingerprint, row order and model ID; it must match the current data.

## Split and provenance checks

A/B/C/F use a complete-system manifest keyed by `raw_case_id`, each containing 32 variations. Task B's sampling 3 runner and its consumers reuse it. D creates one row per system before splitting. E groups trajectory indices by `index // 32` before splitting. `tests/test_case_split.py` and `tests/test_task_e_sampling.py` cover group separation. Preliminary Task E adapter runs and the old Task C fraction objective are outside the main A–F result path.

The raw shards, completion tables, foundation weights and some frozen embeddings are untracked. The repository tracks 1,680 historical generated result and run files (about 182 MB). They are grouped by task and dataset under `results/`; [the path map](artifact_path_map.csv) records their original locations and [the merge map](artifact_merge_map.csv) records the final consolidation. These include provenance used by the research log. Some historical configs contain absolute paths and cannot be replayed verbatim. The reported E iTransformer and TCN final runs both use a three-epoch continuation from an earlier best checkpoint. `scripts/task_e.sh` preserves the base-and-resume sequence; confirm the resulting best epoch and metrics against the saved configs. Manuscript table/figure numbers and final citation are not present here, so this map uses task and result names.
