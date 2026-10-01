# Experiment Plan

## Goal

Build a clean, reproducible Panda forecasting pipeline on the local three-body dataset under `experiments/foundation_models/`, without modifying the main project.

## MVP

### Data

- source dataset: `dataset_v2/`
- local result shards: `0..4`
- representation: `orbital15`
- export length: `640`
- zero-shot / pretrained context length: `512`
- prediction length: `128`
- split seed: `42`
- default export: all locally available complete cases

### Experiments

1. `Exp-0 zero-shot`
   - model: `GilpinLab/panda-72M`
   - no finetune
   - evaluate directly on exported test Arrow
2. `Exp-1 finetune-small`
   - initialization: HF pretrained Panda prediction checkpoint
   - exact 72M-compatible config, 2 CPU steps by default
3. `Exp-2 scratch-small`
   - same data path
   - 349K parameter model, 5 CPU steps by default

### Required outputs

- logs under `outputs/logs/`
- checkpoints under `outputs/checkpoints/`
- eval CSVs and saved forecasts under `outputs/eval/`
- figures under `outputs/figures/`
- top-level summary in `reports/results_summary.md`

## Extended Matrix

Once MVP runs:

1. initialization
   - zero-shot
   - pretrained finetune
   - scratch
2. representation
   - `orbital15`
   - `raw_state18`
   - `geom9`
3. context length
   - `128`
   - `256`
   - `512`
4. horizon
   - `64`
   - `128`
   - `256` after exporting longer sequences than the current `640`

The matrix is intentionally sparse. Not every representation needs every context/horizon combination during the first pass.

## Secondary Probe

Optional after the main forecasting path works:

- use case-level final stability fraction derived from `time_unbound`
- extract simple features from Panda outputs:
  - per-case forecast residual summaries
  - pooled hidden embeddings if readily exposed
- train a lightweight probe:
  - logistic regression or linear SVM

This stays secondary and should not block forecasting.
