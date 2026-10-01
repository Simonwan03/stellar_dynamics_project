# Final sampling-3 system split

`common_case_split_sampling_3_g32_h10000_r70-15-15_seed42.csv` is a byte-for-byte copy of the manifest saved with the final Task B prefix 64, 128, 256, and 512 runs. All four local copies had SHA-256 `24f8f71f3907e39ba1778f506cea9d088a75d638e41d3bea0466de52a0a58f1a`.

It assigns 4,998 complete `raw_case_id` systems: 3,498 train, 750 validation, and 750 test. Each system contains 32 variations. The adjacent JSON records the saved split settings. The original sampling-3 completion table on this machine has SHA-256 `a3ae2b34c0305d8c230401bc15029fdc9db1aa2e1c85303e1c5bf7bd94671216`.

Use this manifest for paper Task B/C/F reproduction. `tasks.common.case_split.ensure_case_split_manifest` validates its system IDs and label-derived stable fractions against the supplied completion table before use. It should not be regenerated or overwritten for an exact paper comparison. Task A uses sampling 2 and needs its separate system split; Task D and E have distinct system-level split implementations.

This CSV contains system-level target counts and split assignments derived from the original dataset. Review its release status before publishing the repository.
