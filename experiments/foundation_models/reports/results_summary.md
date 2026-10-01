# Results Summary

## unbound_orbital15_dataset

Final-unbound classifier on original `dataset/`, using `orbital15` prefix length `512`.

Label: `final_unbound = time_unbound < 5000`.

Current run used CPU fallback because CUDA was not visible in this session; the script is GPU-first when CUDA is available.

| split | n | positive_rate | accuracy | precision | recall | f1 | auroc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 28008 | 0.642316 | 0.883605 | 0.975775 | 0.839633 | 0.902599 | 0.954241 |
| val | 3992 | 0.643036 | 0.888277 | 0.974922 | 0.848072 | 0.907083 | 0.951951 |
| test | 8000 | 0.642625 | 0.884375 | 0.968861 | 0.847306 | 0.904016 | 0.950854 |

- output dir: `outputs/unbound_dataset/unbound_orbital15_dataset`
- checkpoint: `outputs/unbound_dataset/unbound_orbital15_dataset/checkpoint_best.pt`
- predictions: `outputs/unbound_dataset/unbound_orbital15_dataset/predictions_test.csv`

## finetune_orbital15_small_test

| prediction_length | mse | mae | smape | n_systems |
| --- | --- | --- | --- | --- |
| 64 | 4.33478e+09 | 4043.69 | 57.3918 | 12 |
| 128 | 5.60501e+09 | 4347.59 | 61.6773 | 12 |

- metrics dir: `/home/wan/stellar_dynamics/experiments/foundation_models/outputs/eval/finetune_orbital15_small_test/metrics`
- figures dir: `outputs/figures/finetune_orbital15_small_test`

## scratch_orbital15_small_test

| prediction_length | mse | mae | smape | n_systems |
| --- | --- | --- | --- | --- |
| 64 | 2.09162e+10 | 16308.8 | 96.461 | 12 |
| 128 | 2.20187e+10 | 16967 | 96.5859 | 12 |

- metrics dir: `/home/wan/stellar_dynamics/experiments/foundation_models/outputs/eval/scratch_orbital15_small_test/metrics`
- figures dir: `outputs/figures/scratch_orbital15_small_test`

## zero_shot_orbital15_small

| prediction_length | mse | mae | smape | n_systems |
| --- | --- | --- | --- | --- |
| 64 | 4.17944e+09 | 3950.24 | 55.5869 | 12 |
| 128 | 6.297e+09 | 4515.44 | 60.377 | 12 |

- metrics dir: `/home/wan/stellar_dynamics/experiments/foundation_models/outputs/eval/zero_shot_orbital15_small/metrics`
- figures dir: `outputs/figures/zero_shot_orbital15_small`
