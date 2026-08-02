# Task A rerun summary (expanded metrics)

本批实验使用 `seed=42`、`70/10/20` system split、`time_horizon=10000`，并将扩展指标写入每个子目录的 `metrics.json`。GPU 运行使用 WSL 中的 RTX 5060。

## Stable-fraction 主实验

| Model | MSE | MAE | R² | Spearman | MSE / binomial noise |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP-small | 0.08553 | 0.20680 | 0.42798 | 0.64393 | 46.05 |
| MLP-base | 0.09157 | 0.21485 | 0.38760 | 0.62934 | 49.30 |
| MLP-large | 0.09377 | **0.20326** | 0.37285 | 0.63198 | 50.49 |
| XGBoost-base | 0.08624 | 0.22388 | 0.42327 | 0.65034 | 46.43 |
| XGBoost-strong | **0.08336** | 0.21765 | **0.44251** | **0.66239** | **44.88** |

## Binary-any 主实验

正类为 32 条 variation 全部在 horizon 内保持稳定，测试集正类比例为 `15.8%`。

| Model | F1 | AUROC | AUPRC | Balanced Acc. | MCC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP-small | 0.43137 | 0.82748 | 0.47315 | 0.64911 | 0.36753 | 0.10756 | 0.04670 |
| MLP-base | **0.45890** | 0.82711 | 0.47549 | **0.67224** | **0.36884** | 0.10977 | 0.05861 |
| MLP-large | 0.42636 | 0.82360 | 0.45381 | 0.64733 | 0.35825 | 0.10928 | 0.05056 |
| XGBoost-small | 0.21053 | 0.82426 | 0.46076 | 0.55617 | 0.23279 | 0.10652 | **0.01183** |
| XGBoost-base | 0.29108 | 0.84806 | 0.48913 | 0.58385 | 0.26830 | 0.10256 | 0.01874 |
| XGBoost-strong | 0.35242 | **0.85299** | **0.50816** | 0.60936 | 0.31476 | **0.10120** | 0.01993 |

## Horizon ablation

| Target / horizon | Model | MSE or F1 | AUROC | AUPRC |
| --- | --- | ---: | ---: | ---: |
| stable_fraction / 5000 | MLP-base | MSE 0.09264 | — | — |
| stable_fraction / 5000 | XGBoost-base | MSE 0.08906 | — | — |
| stable_fraction / 10000 | MLP-base | MSE 0.09157 | — | — |
| stable_fraction / 10000 | XGBoost-base | MSE 0.08624 | — | — |
| binary_any / 5000 | MLP-base | F1 0.46207 | 0.85466 | 0.58668 |
| binary_any / 5000 | XGBoost-base | F1 0.38849 | 0.86887 | 0.60355 |
| binary_any / 10000 | MLP-base | F1 0.45890 | 0.82711 | 0.47549 |
| binary_any / 10000 | XGBoost-base | F1 0.29108 | 0.84806 | 0.48913 |

## Loss ablation (binary-any, MLP-base, horizon=10000)

| Loss | F1 | AUROC | AUPRC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| BCEWithLogitsLoss | 0.45890 | **0.82711** | **0.47549** | 0.10977 | 0.05861 |
| MSE on sigmoid output | 0.42969 | 0.81445 | 0.47010 | **0.10927** | **0.04526** |

## Feature ablation (MLP-base, horizon=10000)

| Features | Regression MSE | Regression MAE | Binary F1 | Binary AUROC | Binary AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| all orbital15 | 0.09157 | 0.21485 | **0.45890** | 0.82711 | 0.47549 |
| no AP/LAN angles | 0.08787 | 0.21887 | 0.30189 | **0.85104** | **0.50852** |
| no ecc/incl | 0.11714 | 0.26340 | 0.19718 | 0.71838 | 0.30933 |
| aratio+mratio+ecc | **0.08751** | 0.22204 | 0.24510 | 0.84400 | 0.48692 |
| aratio+mratio | 0.11588 | 0.26706 | 0.00000 | 0.74106 | 0.33754 |

各目录还保存了 `config.json`、模型 checkpoint、训练日志和测试预测/曲线文件。
