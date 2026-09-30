# Sampling3 fixed-budget phase-ensemble results

This is the first B=512 run on sampling3: (K,T) = (1,512), (2,256), (4,128), (8,64), (16,32).
MAE uses one prediction per base system after averaging the five training seeds and 50 nested phase draws.
Confidence intervals and paired comparisons resample systems (2,000 replicates); K>1 p-values are Holm-adjusted within each family.

## Test-set composition

- Test systems: 750.
- Near MA boundary, abs(ma_margin) < 0.1: 301 systems.
- Far from boundary, abs(ma_margin) >= 0.3: 123 systems.
- The common pre-instability candidate pool had 18,769 eligible system-draws out of 37,500; 472 distinct test systems had at least one eligible draw.

## DeepSets MAE

| Target/control | K=1,T=512 | K=2,T=256 | K=4,T=128 | K=8,T=64 | K=16,T=32 |
|---|---:|---:|---:|---:|---:|
| All-phase target | 0.2148 [0.2027, 0.2278] (n=750) | 0.2030 [0.1892, 0.2170] (n=750) | 0.2055 [0.1906, 0.2190] (n=750) | 0.2157 [0.2016, 0.2294] (n=750) | 0.2286 [0.2125, 0.2429] (n=750) |
| Held-out target | 0.2117 [0.1987, 0.2240] (n=750) | 0.2025 [0.1898, 0.2155] (n=750) | 0.2076 [0.1942, 0.2216] (n=750) | 0.2184 [0.2036, 0.2337] (n=750) | 0.2308 [0.2162, 0.2460] (n=750) |
| Held-out + pre-instability | 0.3119 [0.2943, 0.3293] (n=472) | 0.3102 [0.2921, 0.3267] (n=472) | 0.3100 [0.2920, 0.3279] (n=472) | 0.3101 [0.2936, 0.3283] (n=472) | 0.3134 [0.2961, 0.3324] (n=472) |

## Paired MAE gain relative to K=1

Positive delta MAE means the multi-phase allocation has lower error than K=1.

| DeepSets target/control | K | delta MAE | 95% system-bootstrap CI | Holm p |
|---|---:|---:|---:|---:|
| All-phase | 2 | +0.0118 | [+0.0089, +0.0146] | 0.0020 |
| All-phase | 4 | +0.0093 | [+0.0041, +0.0145] | 0.0020 |
| All-phase | 8 | -0.0009 | [-0.0073, +0.0055] | 0.7901 |
| All-phase | 16 | -0.0138 | [-0.0216, -0.0061] | 0.0050 |
| Held-out | 2 | +0.0092 | [+0.0057, +0.0126] | 0.0020 |
| Held-out | 4 | +0.0041 | [-0.0018, +0.0099] | 0.1679 |
| Held-out | 8 | -0.0067 | [-0.0134, +0.0004] | 0.1219 |
| Held-out | 16 | -0.0191 | [-0.0272, -0.0106] | 0.0020 |
| Held-out + pre-instability | 2 | +0.0017 | [-0.0011, +0.0046] | 0.9315 |
| Held-out + pre-instability | 4 | +0.0019 | [-0.0031, +0.0072] | 1.0000 |
| Held-out + pre-instability | 8 | +0.0019 | [-0.0053, +0.0092] | 1.0000 |
| Held-out + pre-instability | 16 | -0.0014 | [-0.0118, +0.0078] | 1.0000 |

## Interpretation

- For the all-phase target, DeepSets improves at K=2 and K=4; K=2 has the lowest MAE. K=8 is indistinguishable from K=1, while K=16 is worse.
- For the held-out target without the pre-instability restriction, K=2 improves over K=1; K=4 and K=8 do not show a significant gain, and K=16 is worse.
- Under the strict held-out + pre-instability control, gains at K=2, 4, and 8 are small and their paired intervals include zero. The claim that breadth improves before observed disruption is not established in this run.
- The per-trajectory mean-probability baseline favors K=1; its MAE increases as K grows. The observed breadth benefit is therefore specific to the trained set regressor in the standard and held-out settings.
- In the MA subgroups, descriptive breadth gains at K=2, 4, and 8 are larger in the far group than near the boundary. Treat this cautiously: the far group has 123 systems, the near group 301, and no formal between-group interaction test was specified.

Following the first-round gate, do not expand to additional budgets yet: the primary and held-out settings show a K=2 signal, but the strict pre-instability control does not confirm it.

## Files

- config.json: protocol and random streams.
- phase_partition.csv, system_targets.csv: fixed per-system phase roles and targets.
- per_system_predictions.csv, metrics.csv, paired_bootstrap_comparisons.csv: predictions, metrics, and uncertainty.
- figure1_fixed_budget_tradeoff.png, figure2_hard_controls.png, figure3_ma_boundary_breadth_gain.png: result figures.
