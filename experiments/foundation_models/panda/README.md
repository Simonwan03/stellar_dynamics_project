# Historical Panda pilot notes

The material below documents the earlier `dataset_v2`, eight-variation Panda pilot. Its commands assume `cd experiments/foundation_models` and use compatibility launchers under `scripts/`. For current sampling 3 Task B code use `panda/train_panda_task_b_split.py`; Task E uses `tasks.task_e` from the repository root. See the [foundation-model overview](../README.md) and [paper reproduction map](../../../docs/reproduction.md).

This subproject wires the local three-body trajectory dataset into Panda without touching files outside `experiments/foundation_models/`.

## Status

- Audit source repo: completed from local notebooks, labels, saved notebook outputs, and the vendored Panda repo.
- Default dataset choice: `dataset_v2/` with local result shards `results_batch_00..04_sampling_1.hdf5`.
- Default representation: `orbital15`.
- Export/validation: completed with `target=(15, 640)` and Panda windows `past=(512, 15)`, `future=(128, 15)`.
- MVP experiments: zero-shot, 2-step pretrained finetune, and 5-step scratch sanity check have all run end-to-end on the small test subset.
- Panda source: vendored at `vendor/panda/` from `abao1999/panda`, commit `c229e7c8c49cbe294458c44160248bb17856a715`.

## Why `dataset_v2` and `orbital15`

The local repo contains two forecasting-capable trajectory sources:

1. `dataset/`
   - 5 result shards, 40000 trajectories, 5000 perturbation sets, 1001 steps, `time_unbound` max `5000`.
2. `dataset_v2/`
   - label file covers 50 shards, but the local checkout currently has result shards `00..04`.
   - those 5 local shards still give 40000 trajectories and 5000 complete perturbation sets.
   - each trajectory has 2001 steps and a longer target horizon with `time_unbound` max `10000`.

The main representation used here is the 15-channel orbital stack already exercised in `task_c.ipynb`:

- `aratios_complete`: 1
- `mratios_complete`: 2
- `eccs_complete`: 2
- `incs_complete`: 2
- `sin/cos(APs_complete)`: 4
- `sin/cos(LANs_complete)`: 4

Reasons:

- It is the cleanest forecasting-ready representation already used end-to-end in the main repo.
- The saved notebook outputs show a clean tensor `(40000, 200, 15)` for the locally available `dataset_v2` shards.
- It avoids the known NaN-heavy channels `periods_complete` and `tAnos_complete`.
- It keeps the original perturbation-set structure `(5000, 8, T, 15)` that later enables the optional stability probe.

Alternative representations scaffolded in the adapter:

1. `orbital15` (default, highest priority)
2. `raw_state18` = flattened positions + velocities per body
3. `geom9` = pairwise distances, relative speed magnitudes, radial speeds
4. `full38` = richer notebook feature stack with masses, energy, angular momentum, geometry, normalized time

## Directory Layout

```text
experiments/foundation_models/
  README.md
  reports/
  configs/
  scripts/
  adapters/
  data/
    raw_links/
    panda_arrow/
  outputs/
    logs/
    checkpoints/
    eval/
    figures/
  vendor/
    panda/
```

## Environment

This project uses the single locked `stellar` conda environment described in the repository README.

Bootstrap:

```bash
cd experiments/foundation_models
bash scripts/bootstrap.sh
```

What `bootstrap.sh` does:

- syncs the complete hash-locked dependency set into conda env `stellar`
- keeps Panda source isolated at `vendor/panda/`

Override the conda environment with `PANDA_CONDA_ENV=...` if needed.

## End-to-End Commands

Audit:

```bash
cd experiments/foundation_models
PYTHONPATH="$PWD:$PWD/vendor/panda" conda run -n stellar python scripts/audit_repo.py --write
```

Export primary Panda Arrow dataset:

```bash
PYTHONPATH="$PWD:$PWD/vendor/panda" conda run -n stellar python scripts/export_to_panda_arrow.py --config configs/export.yaml --representation orbital15
```

Validate exported dataset with Panda round-trip:

```bash
PYTHONPATH="$PWD:$PWD/vendor/panda" conda run -n stellar python scripts/validate_panda_dataset.py --config configs/export.yaml
```

Run zero-shot:

```bash
PYTHONPATH="$PWD:$PWD/vendor/panda" conda run -n stellar python scripts/run_zero_shot.py --config configs/zero_shot.yaml
```

Run lightweight finetune:

```bash
bash scripts/run_finetune.sh
```

Run lightweight scratch training:

```bash
bash scripts/run_scratch.sh
```

Run evaluation and summary:

```bash
bash scripts/run_eval.sh
```

Run final-unbound classifier on `dataset/` using `orbital15`:

```bash
bash scripts/run_unbound_dataset.sh
```

Run the Panda-backed version. This script uses `conda run --no-capture-output`, so batch logs stream to the terminal:

```bash
bash scripts/run_panda_unbound_dataset.sh --epochs 3 --log-every-batches 20
```

If the current shell cannot see a GPU but you want a CPU sanity run:

```bash
bash scripts/run_unbound_dataset.sh --allow-cpu --epochs 3
```

One-shot pipeline:

```bash
bash scripts/run_all.sh
```

## Split Rule

- If explicit split manifests already exist under the experiment export root, reuse them.
- Otherwise, use a deterministic case-level split with `seed=42`.
- Splits are built on complete 8-trajectory perturbation sets, never on individual trajectories.
- Split manifests are written under `data/panda_arrow/manifests/`.

## MVP Results

The latest small-run summary is in `reports/results_summary.md`.

- zero-shot: `outputs/eval/zero_shot_orbital15_small/`
- finetune checkpoint: `outputs/checkpoints/finetune_orbital15_small-0/checkpoint-final/`
- scratch checkpoint: `outputs/checkpoints/scratch_orbital15_small-0/checkpoint-final/`
- finetune eval: `outputs/eval/finetune_orbital15_small_test/`
- scratch eval: `outputs/eval/scratch_orbital15_small_test/`
- figures: `outputs/figures/{run_name}/`

## Final-Unbound Classification

`configs/unbound_dataset.yaml` defines a GPU-first supervised probe on the original `dataset/` directory:

- input: `orbital15` prefix, default first `512` steps
- label: `final_unbound = time_unbound < max(time_unbound)`; for local `dataset/`, max time is `5000`
- split: deterministic case-level split, `70/10/20`
- model: lightweight temporal Conv1D classifier in PyTorch

Current run status in this environment:

- CUDA was not visible to PyTorch, so the completed run used `--allow-cpu --epochs 3`.
- Full local `dataset/` was used: `40000` trajectories, `5000` complete 8-way cases.
- Outputs are under `outputs/unbound_dataset/unbound_orbital15_dataset/`.
- Test metrics: accuracy `0.8844`, F1 `0.9040`, AUROC `0.9509`.

## Chronos-2 Task B Classification (`prefix=96`)

The full Task B split was encoded with the local Chronos-2 checkpoint and a
frozen-backbone linear probe was trained on the resulting features.

- Split: `125984 / 13984 / 19985` train/validation/test samples, identical to the Panda Task B split.
- Input: `orbital15` with shape `(15, 96)`; Chronos-2 uses six 16-step context patches.
- Feature: the penultimate `[REG]` token (token order is context patches, `[REG]`, masked-output patch), giving `15 x 768 = 11520` features per sample.
- Cache: `159953 x 11520`, stored as `float16` in `outputs/panda_unbound_dataset/chronos2_task_b_prefix96_reg_features.float16.npy`.
- Classifier: one linear layer with `11521` trainable parameters, trained for 20 epochs on CUDA.

Reproduce the linear-probe run from this directory:

```bash
conda run -n stellar python scripts/train_chronos2_task_b_linear.py \
  --feature-cache outputs/panda_unbound_dataset/chronos2_task_b_prefix96_reg_features.float16.npy \
  --prefix-dir ../../results/task_b/sampling_2/task_b_common/prefix_512 \
  --output-dir outputs/panda_unbound_dataset/chronos2_task_b_split_prefix96_reg \
  --epochs 20 --batch-size 2048 --device cuda
```

### Test-set comparison

The positive label is shown in the `target` column; both models use threshold
`0.5` for the thresholded metrics. `n=19985` for every row below.

| Target | Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| Unbound | Chronos-2 REG, prefix 96 | 0.86645 | 0.95000 | 0.84941 | 0.89689 | 0.95076 | 0.97745 |
| Unbound | Panda, prefix 96 | 0.86715 | 0.94161 | 0.85899 | 0.89840 | 0.94545 | 0.97473 |
| Stable | Chronos-2 REG, prefix 96 | 0.86645 | 0.73500 | 0.90331 | 0.81051 | 0.95076 | 0.89956 |
| Stable | Panda, prefix 96 | 0.86715 | 0.74368 | 0.88479 | 0.80812 | 0.94545 | 0.88375 |

Confusion matrices are ordered as `[[TN, FP], [FN, TP]]`:

- Unbound: Chronos-2 `[[5708, 611], [2058, 11608]]`; Panda `[[5591, 728], [1927, 11739]]`.
- Stable: Chronos-2 `[[11608, 2058], [611, 5708]]`; Panda `[[11739, 1927], [728, 5591]]`.

Chronos-2 improves test AUROC by `+0.00531` and stable-target AUPRC by
`+0.01581` over Panda at the same prefix. Panda with the longer `prefix=512`
remains stronger overall (`test AUROC=0.97872`, stable-target `F1=0.89130`).

Validation AUROC is `0.95718` for Chronos-2 and `0.94932` for Panda at
`prefix=96`. Full metrics and predictions are saved under
`outputs/panda_unbound_dataset/chronos2_task_b_split_prefix96_reg/`.

## Current Audit Snapshot

- Local repo code is notebook-heavy; the reusable data logic comes primarily from `read_data.ipynb`, `task_b.ipynb`, and `task_c.ipynb`.
- `task_b.ipynb` saved outputs show NaN-heavy channels:
  - `periods_complete`: `16847 / 40000`
  - `tAnos_complete`: `14509 / 40000`
  - `positions_complete`: `254 / 40000` within the first `T=100`
- `task_c.ipynb` saved outputs show the local `dataset_v2` five-shard subset already cleaned to:
  - trajectories: `40000`
  - cases: `5000`
  - case tensor: `(5000, 8, 200, 15)`
  - trajectory target tensor: `(5000, 8, 9)`

Detailed paths, shapes, and prioritization are recorded in `reports/data_audit.md`.

## Known Issues

- The main repo does not expose its notebook data logic as importable Python modules, so the adapter re-implements notebook-equivalent logic inside this isolated subproject instead of modifying the original notebooks.
- Local `dataset_v2/` labels reference 50 shards, but only 5 result shards are currently present in the repo checkout. All scripts default to the locally available subset unless you add more shards.
- GPU probing may be blocked by the current OS sandbox; launch scripts therefore default to CPU-safe settings. The 72M finetune sanity run took about 75 seconds for 2 CPU steps.
- Panda’s training script has no built-in validation loop; validation and test evaluation are run explicitly after training.
