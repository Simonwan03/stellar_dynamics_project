# Manuscript reproduction map

Run commands from the repository root. The sampling 3 data directory must contain the high-resolution HDF5 shards and matching completion table. Those files are absent here. The `dataset_v3` symlink targets a developer machine, so supply sampling 2 data separately. Saved result `config.json` files record exact reported invocations; current CLI defaults can differ. Training entry points also evaluate validation and test unless an evaluator is named below.

| Task | Current pipeline | Reported settings and result provenance |
| --- | --- | --- |
| A: system stable fraction | `tasks.task_a` loads 32 initial-condition variations per system. `tasks/common/case_split.py` assigns complete `raw_case_id` systems. | Sampling 2 MLP, XGBoost and analytic baselines; `--target-mode stable_fraction`. See [Task A research log](../readme.md). `binary_any` is a different target. No sampling 3 Task A result config is present. |
| B: short-prefix stability | `scripts/run_task_b_highres.py` loads sampling 3 orbital15 prefixes. `experiments/panda_threebody/scripts/train_spock_inspired_task_b_xgboost.py` builds summary features; the same directory contains Panda and Chronos-2 encoder/head drivers. | Prefixes 64/128/256/512; horizon 10000; [prefix 128 neural config](../results/task_b_sampling3_prefix128/config.json) and [Summary+XGBoost config](../results/task_b_sampling3_prefix128_stats_xgboost/run_config.json). `scripts/evaluate_task_b_highres_two_level.py` computes system stable-fraction metrics from trajectory predictions. Frozen embeddings and weights are external. |
| C: survival curve | `tasks.task_c.survival` constructs censored discrete hazards from complete systems and aligned Task B prefix data. | Sampling 3 prefix 128, 20 bins from 1 to 10000, seven readout horizons, 3498/750/750 system split. [Raw](../results/task_c_sampling3_prefix128_raw/config.json), [Panda](../results/task_c_sampling3_prefix128_panda/config.json), [Chronos-2](../results/task_c_sampling3_prefix128_chronos2/config.json). `tasks.task_c` implements the older fraction objective. |
| D: finite-time divergence | `tasks.task_d` calls `tasks/task_d/quantile.py`. It derives paired semi-major-axis crossing times and one initial-condition vector per system. | Default exp(5) threshold, eight valid pairs, interpolated system median target (the current entry does not fit multiple quantiles), 70/15/15 split; constant, XGBoost and MLP. Evaluation, bootstrap and split CSV are written during training. `tasks/task_d/pipeline.py` is historical. |
| E: trajectory forecasting | `tasks.task_e.prepare` builds a completion-filtered, train-only scaled, one-start cache; `tasks.task_e` evaluates E1 one-shot and E2 four-block recursive predictions. | Sampling 3 first 512 to next 128, orbital15, system-balanced metrics. Saved configs: [Chronos-2](../runs/task_e_sampling3_first512_chronos2/config.json), [Panda](../runs/task_e_sampling3_first512_panda/config.json), [Toto-2](../runs/task_e_sampling3_first512_toto2/config.json), [iTransformer](../runs/task_e_sampling3_first512_itransformer_ep25/config.json), [TCN](../runs/task_e_sampling3_first512_tcn_ep25_continued/config.json). |
| F: partial observation | `tasks.task_f` trains masked-prefix TCN/attention; `scripts/run_task_f_summary_xgboost.py` evaluates summary XGBoost; `scripts/run_task_f_mixed_mask_benchmark.py` compares mask regimes. Task E `--mask-regime` covers masked forecasting. | Sampling 3 prefix 256, orbital15 plus observed flag, horizon 10000. [Random mask](../results/task_f_masked_prefix/config.json), [mixed mask](../results/task_f_mixed_mask_benchmark_seed42/config.json), [masked iTransformer forecasting](../runs/task_e_sampling3_first512_itransformer_masked_mixed_r50/config.json). Keep clean, random, block and mixed protocols separate. |

## Training and evaluation commands

The following are verified entry points, not claims that the external data or weights are available. Consult each linked config to reproduce a particular model/table row exactly; pass its options to the matching entry point. These commands write evaluation metrics during training.

```bash
# A: sampling 2 stable fraction
python -m tasks.task_a --dataset-dir dataset_v3 --target-mode stable_fraction --models mlp,xgboost --output-dir results/reproduce_task_a

# B: sampling 3 prefix 128 MLP/CNN/TCN
python scripts/run_task_b_highres.py --dataset-dir dataset_sampling_3_prefix1024 --prefix-len 128 --models mlp,cnn,tcn --output-dir results/reproduce_task_b

# C: sampling 3 raw/summary survival heads
python -m tasks.task_c.survival --dataset-dir dataset_sampling_3_prefix1024 --sampling-id 3 --prefix-len 128 --hazard-t-min 1 --hazard-t-max 10000 --hazard-bins 20 --variation-target full32 --epochs 20 --models baseline,stats,cnn,tcn,gru,attention --output-dir results/reproduce_task_c

# D: median divergence-time regression
python -m tasks.task_d --dataset-dir dataset_sampling_3_prefix1024 --models constant,xgboost,mlp --output-dir results/reproduce_task_d

# E: prepare one-start cache, then train/evaluate E1 and E2
bash scripts/reproduce/task_e.sh chronos2 dataset_sampling_3_prefix1024

# F: random-mask prefix classification
python -m tasks.task_f --dataset-dir dataset_sampling_3_prefix1024 --prefix-len 256 --mask-ratios 0,25,50,75 --models tcn,attention --output-dir results/reproduce_task_f
```

For the reported sampling 3 Summary+XGBoost row, use `python experiments/panda_threebody/scripts/train_spock_inspired_task_b_xgboost.py --dataset-dir dataset_sampling_3_prefix1024 --sampling-id 3 --prefix-dir results/task_b_sampling3_prefix512 --prefix-steps 128 --output-dir results/reproduce_task_b_stats_xgboost`. Task B's separate system-level evaluator reads the run's prediction files; use `python scripts/evaluate_task_b_highres_two_level.py --help` to select inputs and outputs. Frozen Panda and Chronos-2 B/C rows need aligned embeddings prepared by scripts in `experiments/panda_threebody/scripts/`. Cache metadata records label fingerprint, row order and model ID; it must match the current data.

## Split and provenance checks

A/B/C/F use a complete-system manifest keyed by `raw_case_id`, each containing 32 variations. Task B's sampling 3 runner and its consumers reuse it. D creates one row per system before splitting. E groups trajectory indices by `index // 32` before splitting. `tests/test_case_split.py` and `tests/test_task_e_sampling.py` cover group separation. The older Task G boundary study, preliminary Task E adapter runs, and old Task C fraction objective are outside the main A–F result path.

The raw shards, completion tables, foundation weights and some frozen embeddings are untracked. The repository still tracks historical generated results and runs (1,680 files, about 182 MB); these include provenance used by the research log, and were not removed without a reviewed replacement archive. Some historical configs contain absolute paths and cannot be replayed verbatim. The reported E iTransformer and TCN final runs both use a three-epoch continuation from an earlier best checkpoint. `scripts/reproduce/task_e.sh` preserves the base-and-resume sequence; confirm the resulting best epoch and metrics against the saved configs. Manuscript table/figure numbers and final citation are not present here, so this map uses task and result names.
