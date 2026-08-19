# Task E 预测窗口误差实验结果

> 历史记录：本文对应旧的 `64 -> 32` Panda adapter 实验。当前 Task E 主协议已经
> 改为 E1 `512 -> 128` one-shot 与 E2 四段 recursive rolling；本文结果不属于新的
> E1/E2 benchmark。

## 实验目的

固定输入 prefix 长度为 64，预测未来 32 个时间步，使用 orbital15 表征，检查预测误差是否随预测 horizon 增加。

本报告记录 Panda adapter 的 GRU/MLP decoder 对照实验，以及 TimesFM、Chronos 和 Panda zero-shot 结果。

## 数据与窗口

- 数据：sampling_2
- 表征：orbital15，共 15 个特征
- 输入长度：64
- 预测长度：32
- stride：20
- 每个 32-variation 系统最多随机选择 2 条轨迹
- 每条选中轨迹只保留开头的 2 个 sequential windows
- 随机种子：42
- 使用完整有效窗口，不再设置全局 train/validation/test window 上限

由于完成性检查和有限值检查，最终有效样本数低于理论上的 20,000 个 window：

| split | 轨迹数 | windows |
|---|---:|---:|
| train | 6,116 | 11,777 |
| validation | 1,301 | 2,522 |
| test | 1,319 | 2,556 |
| total | 8,736 | 16,855 |

## 模型配置

- Backbone：Panda-72M
- Panda backbone：冻结
- Adapter decoder：GRU
- GRU 层数：2
- Adapter hidden dimension：128
- Batch size：32
- Adapter learning rate：1e-4
- Weight decay：1e-4
- Gradient clipping：1.0
- 训练设备：NVIDIA A10G
- 总参数量：72,009,483
- 可训练参数量：419,471

### 训练轮数说明

初始训练先完成了 epoch 1–4，之后从 epoch 4 的最佳 checkpoint 继续训练 20 个 epoch，因此最终日志的全局 epoch 为 5–24。

这意味着本次实际执行的是“额外训练 20 个 epoch”，而不是从头开始总共限制为 20 个 epoch。最终模型使用最后阶段产生的最佳 checkpoint。

## 总体指标

最终测试集指标：

| metric | value |
|---|---:|
| MSE（标准化） | 0.341675 |
| MAE（标准化） | 0.378713 |
| RMSE（标准化） | 0.584530 |
| MSE（原始尺度） | 0.152961 |
| MAE（原始尺度） | 0.238729 |
| RMSE（原始尺度） | 0.391102 |
| sMAPE（transformed） | 56.6381% |
| Spearman（transformed） | 0.822295 |

最终验证集的标准化 MSE 为 0.337146，原始尺度 MSE 为 0.150160。

## 误差随预测步的变化

下面的数值来自测试集，误差是在 15 个 orbital15 特征上平均得到的。

| 预测步 | MSE（标准化） | MAE（标准化） | MSE（原始尺度） |
|---:|---:|---:|---:|
| 1 | 0.2187 | 0.2935 | 0.0985 |
| 4 | 0.2726 | 0.3274 | 0.1232 |
| 8 | 0.3148 | 0.3589 | 0.1421 |
| 12 | 0.3426 | 0.3783 | 0.1549 |
| 16 | 0.3432 | 0.3825 | 0.1540 |
| 20 | 0.3696 | 0.3982 | 0.1650 |
| 24 | 0.3751 | 0.4033 | 0.1671 |
| 28 | 0.3860 | 0.4109 | 0.1724 |
| 32 | 0.4128 | 0.4231 | 0.1800 |

### 测试集趋势统计

- 标准化 MSE：0.2187 增加到 0.4128，约为第 1 步的 1.89 倍。
- 原始尺度 MSE：0.0985 增加到 0.1800，约为第 1 步的 1.83 倍。
- 标准化 MSE 有 29/31 个相邻步长不下降。
- 标准化 MSE 与 horizon 的 Spearman 相关系数为 0.9974。
- 标准化 MSE 的线性斜率约为每步 0.00486。

验证集趋势也一致：

- 标准化 MSE：0.2042 增加到 0.4462，约为第 1 步的 2.19 倍。
- 原始尺度 MSE：0.0922 增加到 0.1847，约为第 1 步的 2.00 倍。
- 标准化 MSE 有 28/31 个相邻步长不下降。
- 标准化 MSE 与 horizon 的 Spearman 相关系数为 0.9971。

## 结论

误差随预测窗口长度明显增加。它不是严格意义上的逐步单调递增，但总体趋势非常强：第 32 步的误差大约是第 1 步的 1.8–2.2 倍。

测试集标准化 MSE 的局部下降主要出现在第 14–16 步附近；原始尺度 MSE 还在第 20–22 步附近出现小幅波动。这些局部波动不改变整体的长期递增趋势。

因此，在当前固定 64 步输入、32 步输出的设置下，模型存在明显的长 horizon 误差累积或预测漂移问题。

## MLP decoder 对照实验

在完全相同的 orbital15、prefix=64、pred=32、采样和数据划分下，使用 Panda adapter + MLP decoder 进行对照。

- decoder：MLP
- hidden dimension：128
- Panda backbone：冻结
- learning rate：1e-4
- 最多训练：20 个 epoch
- 选用 checkpoint：第 15 轮
- validation windows：2,522
- test windows：2,556

第 15 轮 checkpoint 的测试集指标：

| metric | value |
|---|---:|
| MSE（标准化） | 0.290387 |
| MAE（标准化） | 0.322718 |
| RMSE（标准化） | 0.538875 |
| MSE（原始尺度） | 0.128972 |
| MAE（原始尺度） | 0.203784 |
| RMSE（原始尺度） | 0.359126 |
| sMAPE（transformed） | 47.8532% |
| Spearman（transformed） | 0.853114 |

MLP 相比之前的 GRU decoder 测试集标准化 MSE 下降约 15.0%，说明在固定输出长度 32 的任务上，MLP 比当前 GRU decoder 更适合直接优化这个预测窗口。需要注意：此前 GRU 的最终 checkpoint 是全局 epoch 24，而本次按要求选择了 MLP 的 epoch 15，因此这不是完全相同训练轮数的严格对照。

MLP 的 horizon MSE-z 从第 1 步的 0.0636 增加到第 32 步的 0.4214；第 1 到第 32 步约增加 6.62 倍。它的短 horizon 精度明显优于 GRU，但第 32 步略高于 GRU 的 0.4128，说明 GRU 的长 horizon 误差增长更慢。

第 15 轮 MLP 结果文件：

- 逐步 horizon 指标：runs/task_e_orbital15_ctx64_pred32_panda_mlp_2var2win_epoch15/horizon_metrics.csv
- 测试集趋势汇总：runs/task_e_orbital15_ctx64_pred32_panda_mlp_2var2win_epoch15/test_horizon_trend.json
- 测试集总体指标：runs/task_e_orbital15_ctx64_pred32_panda_mlp_2var2win_epoch15/test_metrics.json
- 训练日志：runs/task_e_orbital15_ctx64_pred32_panda_mlp_2var2win/train_log.csv

## 结果文件

- 逐步 horizon 指标：runs/task_e_orbital15_ctx64_pred32_panda_gru_2var2win_earlystop/horizon_metrics.csv
- 测试集趋势汇总：runs/task_e_orbital15_ctx64_pred32_panda_gru_2var2win_earlystop/test_horizon_trend.json
- 验证集趋势汇总：runs/task_e_orbital15_ctx64_pred32_panda_gru_2var2win_earlystop/val_horizon_trend.json
- 测试集总体指标：runs/task_e_orbital15_ctx64_pred32_panda_gru_2var2win_earlystop/test_metrics.json
- 训练日志：runs/task_e_orbital15_ctx64_pred32_panda_gru_2var2win_earlystop/train_log.csv
