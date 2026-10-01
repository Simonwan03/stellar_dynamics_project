# Reproducibility audit for public release

This is a status report, not a claim that every result can be rerun from a fresh clone. The [reproduction map](reproduction.md) names current entry points and saved result configs. The checks below were performed in this workspace on 2026-10-01. No training job was rerun for this audit.

## Current state by paper task

| Task | Available here | Remaining requirement for an independent reproduction |
| --- | --- | --- |
| A: system stable fraction | Code, saved sampling-2 outputs, and system-split implementation | The local `dataset_v3` symlink is broken. Supply the sampling-2 initial-condition HDF5 and matching completion table; verify the exact reported split and model settings from saved configs. |
| B: short-prefix stability | Sampling-3 data is present locally; four prefix manifests are byte-identical and the canonical system split is now in [`configs/splits/`](../configs/splits/README.md). The MLP/CNN/TCN runner accepts `--split-manifest`. | Raw data is untracked and has no verified public download location. Summary+XGBoost and foundation-model heads require their separate preprocessing, pretrained weights or embedding caches, and saved row-specific settings. |
| C: survival curve | Sampling-3 raw/Panda/Chronos-2 configs and a CLI for the raw survival models; the same committed system split can be supplied. | Frozen embedding caches and foundation weights are external. Saved configs include paths to local caches; reconstruct and validate their label fingerprints and row order. |
| D: divergence time | Sampling-3 data, saved metrics, and a runnable median-decorrelation entry point. | Confirm which saved quantile/sensitivity runs correspond to final manuscript rows. The current main CLI targets a median regression; the paper task name refers more broadly to quantile prediction. Do not change the target until this is resolved. |
| E: trajectory forecasting | Sampling-3 data, saved E1/E2 configs, and `scripts/task_e.sh` for the main comparison models. | Pretrained foundation weights must be available locally or downloads explicitly enabled; GPU capacity and a clean dependency installation remain unverified. Confirm the saved base-plus-resume checkpoint sequence for iTransformer and TCN. |
| F: masked-prefix robustness | Sampling-3 data, a task CLI, comparison scripts, and the committed B/C/F system split. | Run each reported mask protocol separately and match its saved config; clean, random, block, and mixed regimes are distinct experiments. |

## Verified in this workspace

- The sampling-3 directory resolves to 40 HDF5 shards; each checked shard has `times_complete` shape `(4000, 1024)`. Its completion table SHA-256 is `a3ae2b34c0305d8c230401bc15029fdc9db1aa2e1c85303e1c5bf7bd94671216`.
- The four final Task B prefix runs have identical split CSVs, SHA-256 `24f8f71f3907e39ba1778f506cea9d088a75d638e41d3bea0466de52a0a58f1a`. The copy in `configs/splits/` validates against the sampling-3 completion table and contains 3,498 training, 750 validation, and 750 test systems.
- `python -m tasks.common.preflight --task B` passes here. `--task A` fails because the sampling-2 symlink target is unavailable. This is an input availability failure, not a Task A model failure.
- The suite previously passed 105 tests after the directory migration; unit tests do not establish equality of full retrained paper results.

## Release blockers and next actions

1. **Data access:** publish or document an authorized sampling-2 and sampling-3 data source, or publish a deterministic generation procedure. Do not claim a fresh clone can reproduce training until this exists.
2. **Clean installation:** `environment.yml`, `requirement.txt`, and `requirements.lock` now define one Linux x86_64 / Python 3.12 / CUDA 13.0 path. Dependency resolution and a hash-enforced sync dry run pass, but a clean installation and GPU inference on another machine remain unverified.
3. **Exact model provenance:** record accessible checkpoint or model IDs, versions, and hashes for every final foundation-model row. Saved configs alone do not supply the weights. Verify Task D's final target definition against the manuscript before changing code.
4. **Paper mapping:** add final table/figure numbers and expected metric values once the manuscript is frozen. The repository currently maps by task and run name, so readers cannot yet locate each paper cell unambiguously.
5. **Git payload:** many historical result files remain intentionally tracked for provenance. Review the final public Git size and decide which compact metrics/configs to keep before publication; preserve an archive or path map for any relocated artifacts.

The canonical split CSV includes system-level target counts derived from the data. Confirm that this derived artifact may be released with the paper before publishing it.
