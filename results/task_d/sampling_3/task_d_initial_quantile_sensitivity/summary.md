# Task D label-threshold sensitivity

This ablation varies the crossing threshold in the Task D label definition,
`G = exp(k)`, while keeping the dataset, seed (`42`), split ratios, feature
representation (`summary`: mean/std/min/max over the 16 nominal variations),
and models (`constant`, `gbr`) fixed. Crossing times use log-growth
interpolation and a system requires at least 8 valid pair crossings.

Command:

```bash
conda run --no-capture-output -n stellar python -m tasks.task_d.sensitivity \
  --dataset-dir dataset_sampling_3_prefix1024 \
  --threshold-powers 4 5 6 \
  --models constant,gbr \
  --output-dir results/task_d/sampling_3/task_d_initial_quantile_sensitivity
```

## Results

| Threshold | Systems | Target P25 | Target P50 | Target P75 | Model | Mean pinball ↓ | Central 50% coverage | Log interval width ↓ | P50 log-MAE ↓ | P50 Spearman ↑ | Factor-2 accuracy ↑ |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| e⁴ | 4997 | 0.9544 | 3.4319 | 9.1262 | Constant | 0.4705 | 0.5093 | 2.2646 | 1.1391 | — | 29.87% |
| e⁴ | 4997 | 0.9544 | 3.4319 | 9.1262 | GBR | **0.3227** | 0.4827 | **1.3530** | **0.7559** | 0.6589 | **54.13%** |
| e⁵ | 4996 | 0.9113 | 3.4091 | 11.2786 | Constant | 0.5280 | 0.5100 | 2.5218 | 1.2777 | — | 24.83% |
| e⁵ | 4996 | 0.9113 | 3.4091 | 11.2786 | GBR | **0.3611** | 0.4780 | **1.5320** | **0.8475** | 0.6686 | **51.27%** |
| e⁶ | 4988 | 0.8811 | 3.3414 | 12.0871 | Constant | 0.5703 | 0.5013 | 2.6254 | 1.3665 | — | 24.73% |
| e⁶ | 4988 | 0.8811 | 3.3414 | 12.0871 | GBR | **0.3930** | 0.4920 | **1.6365** | **0.9139** | 0.6810 | **48.13%** |

The corresponding train/validation/test counts are 3497/750/750 for e⁴,
3497/750/749 for e⁵, and 3491/749/748 for e⁶. The system membership is not
perfectly identical across thresholds because of the minimum-valid-pair
filter, so metric trends should be interpreted as threshold sensitivity rather
than a strictly paired benchmark.

GBR remains clearly better than the constant quantile predictor at all three
thresholds. Increasing the threshold makes the target distribution wider in
the upper tail and steadily worsens GBR absolute-error and pinball metrics,
while central interval coverage stays close to the nominal 50%. The full
metrics and predictions are in [`e4/`](e4/), [`e5/`](e5/), and [`e6/`](e6/).
