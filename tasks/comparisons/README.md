# Cross-task comparisons

These studies compare Tasks A, B, and C or explore observation budgets. They
are auxiliary analyses, separate from the Task A–F manuscript reproduction
entry points in `scripts/`.

| Module | Purpose | Data |
| --- | --- | --- |
| `fixed_budget_sampling3.py` | Compare system-level phase-ensemble regression with trajectory-averaged classification at a fixed stored-point budget | sampling 3 |
| `factorial_sampling3.py` | Extend the budget study across numbers of phase pairs and prefix lengths | sampling 3 |
| `evaluate_common_target.py` | Evaluate saved A/B/C predictions on the same system stable-fraction target at `t=10000` | sampling 2 |
| `future_only_ablation.py` | Analyze at-risk trajectory/prefix predictions using the saved common-target comparison | sampling 2 |

Run a module from the repository root, for example:

```bash
python -m tasks.comparisons.evaluate_common_target --help
python -m tasks.comparisons.fixed_budget_sampling3 --help
```

Comparison outputs are under `results/comparisons/`, grouped by dataset.
The [merge map](../../docs/artifact_merge_map.csv) records their former paths.
