# Task A 实验与 Ablation 汇总

最新一批使用扩展 metrics 重新运行的结果见
[`metrics_augmented/summary.md`](metrics_augmented/summary.md)；本文件保留此前主实验汇总。
新增的细粒度 horizon 网格见
[`metrics_augmented/horizon_grid/summary.md`](metrics_augmented/horizon_grid/summary.md)。
新增的短 horizon stable-fraction 网格见
[`metrics_augmented/horizon_grid_short/summary.md`](metrics_augmented/horizon_grid_short/summary.md)。

## 实验口径

- 数据：`dataset_v3`，保留 `4998` 个有效且完整的 32-variation systems。
- 输入：每组 32 条 variation 的第一个初始条件；角变量展开为 sin/cos。
- split：`70% train / 10% validation / 20% test`，`seed=42`。
- MLP：最多 50 epochs，validation early stopping，CPU 结果用于横向 ablation。
- XGBoost：`hist` tree method。
- 表中二分类正类表示 32 条 variation 全部在 horizon 内保持稳定。

## Stable-fraction 回归（horizon=10000）

| Model | Test MSE | RMSE | MAE | R² |
| --- | ---: | ---: | ---: | ---: |
| MLP-small | 0.08489 | 0.29135 | **0.20280** | 0.43229 |
| MLP-base | 0.09184 | 0.30305 | 0.21867 | 0.38579 |
| MLP-large | 0.08799 | 0.29662 | 0.20930 | 0.41156 |
| XGBoost-base | 0.08624 | 0.29366 | 0.22388 | 0.42327 |
| XGBoost-strong | **0.08336** | **0.28872** | 0.21765 | **0.44251** |

XGBoost-strong 是该设置下整体最好的回归模型；MLP-small 的 MAE 最低。

当前代码会在每个 Task A 模型的 `metrics.json` 中保留上述主指标，并附加：
回归的 median absolute error、explained variance、Pearson/Spearman、bias、误差容忍率、
目标比例分箱误差和二项采样噪声比；二分类的 balanced accuracy、specificity、FPR/FNR、
MCC、TN/FP/FN/TP、Brier、log loss、ECE/MCE。旧字段保持不变，因此既有汇总表仍可直接比较。

## Binary-any 分类（horizon=10000）

正类比例为 `790/4998 = 15.81%`。以下是 CPU 横向结果：

| Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP-small | **0.855** | 0.562 | **0.373** | **0.449** | 0.836 | 0.481 |
| MLP-base | 0.852 | 0.550 | 0.348 | 0.426 | 0.825 | 0.481 |
| MLP-large | 0.851 | 0.545 | 0.342 | 0.420 | 0.823 | 0.467 |
| XGBoost-small | 0.850 | **0.625** | 0.127 | 0.211 | 0.824 | 0.461 |
| XGBoost-base | 0.849 | 0.564 | 0.196 | 0.291 | 0.848 | 0.489 |
| XGBoost-strong | 0.853 | 0.580 | 0.253 | 0.352 | **0.853** | **0.508** |

GPU 顺序主实验也已完成，输出在 `main_binary_cuda/`：MLP-base 的
`F1=0.459`、`AUROC=0.827`、`AUPRC=0.475`；XGBoost 结果与 CPU 版本一致。

## Horizon ablation

以下结果使用相同的 `seed=42`、`70/10/20` system split 和 `orbital15` 输入；
100–2000 的 stable-fraction 行是本批 CPU 重新运行的结果。

| Target | Horizon | Stable-fraction mean | MLP-base | XGBoost-base |
| --- | ---: | ---: | --- | --- |
| stable_fraction | 100 | 0.745861 | MSE 0.051570, MAE 0.146435, R² 0.599556 | MSE 0.049802, MAE 0.155947, R² 0.613285 |
| stable_fraction | 200 | 0.682454 | MSE 0.063825, MAE 0.179518, R² 0.562286 | MSE 0.059510, MAE 0.178483, R² 0.591881 |
| stable_fraction | 500 | 0.587529 | MSE 0.076093, MAE 0.201114, R² 0.536037 | MSE 0.074599, MAE 0.211459, R² 0.545143 |
| stable_fraction | 1000 | 0.513718 | MSE 0.088373, MAE 0.216604, R² 0.482520 | MSE 0.086658, MAE 0.234104, R² 0.492558 |
| stable_fraction | 2000 | 0.445616 | MSE 0.090029, MAE 0.224702, R² 0.469801 | MSE 0.089360, MAE 0.237237, R² 0.473743 |
| stable_fraction | 5000 | — | MSE 0.08933, R² 0.44595 | MSE 0.08906, R² 0.44758 |
| stable_fraction | 10000 | — | MSE 0.09184, R² 0.38579 | MSE 0.08624, R² 0.42327 |
| binary_any | 5000 | — | F1 0.465, AUROC 0.854, AUPRC 0.571 | F1 0.388, AUROC 0.869, AUPRC 0.604 |
| binary_any | 10000 | — | F1 0.426, AUROC 0.825, AUPRC 0.481 | F1 0.291, AUROC 0.848, AUPRC 0.489 |

当前 completion table 的 `time_unbound` 最大值为 `10000`，代码会拒绝超过记录范围的 horizon；此前生成的
`fraction_h20000_*` 目录仅是边界检查产物，不纳入正式结果。

## Feature ablation（MLP-base，horizon=10000）

### Stable-fraction

| Features | Dimensions | MSE | MAE | R² |
| --- | ---: | ---: | ---: | ---: |
| all orbital15 | 15 | 0.09184 | 0.21867 | 0.38579 |
| no AP/LAN angles | 7 | **0.08508** | **0.21588** | **0.43098** |
| no ecc/incl | 9 | 0.11721 | 0.26162 | 0.21610 |
| aratio+mratio+ecc | 5 | 0.08746 | 0.22028 | 0.41509 |
| aratio+mratio | 3 | 0.11527 | 0.27064 | 0.22911 |

### Binary-any

| Features | Dimensions | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: |
| all orbital15 | 15 | 0.426 | 0.825 | 0.481 |
| no AP/LAN angles | 7 | 0.423 | **0.858** | **0.520** |
| no ecc/incl | 9 | 0.110 | 0.721 | 0.300 |
| aratio+mratio+ecc | 5 | 0.327 | 0.849 | 0.505 |
| aratio+mratio | 3 | 0.000 | 0.739 | 0.334 |

## Loss ablation（binary-any，MLP-base）

| Loss | Accuracy | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: |
| BCEWithLogitsLoss | 0.852 | 0.426 | **0.825** | **0.481** |
| MSE on sigmoid output | **0.855** | **0.444** | 0.811 | 0.457 |

BCE 的概率排序指标更好；MSE 在固定 0.5 threshold 下的 F1 略高。由于正类仅约
15.8%，后续正式二分类实验建议增加 `pos_weight` 或在 validation set 上调分类阈值。

## Output locations

每组实验的 `config.json`、`metrics.json`、模型和图保存在本目录下对应的子目录，
例如：

- `main_fraction_cuda/`
- `main_binary_cuda/`
- `fraction_feat_no_angles/`
- `binary_feat_no_angles/`
- `fraction_h5000_mlp_base/`
