# Task E Forecasting Pipeline

Task E 是统一的“过去窗口预测未来窗口”多步时间序列预测任务。正式入口为：

```bash
python -m tasks.task_e --help
```

主要代码位于：

- `tasks/task_e/main.py`：CLI、训练、推理和实验产物保存
- `tasks/task_e/data.py`：HDF5 数据读取、特征变换、数据切分、scaler 和滑动窗口
- `tasks/task_e/models.py`：TCN、PatchTST 和简化 iTransformer
- `tasks/task_e/transformer.py`：Transformer forecaster
- `tasks/task_e/foundation.py`：TimesFM、Chronos 和 Panda 加载与 zero-shot 推理
- `tasks/task_e/panda_adapter.py`：Panda temporal adapter
- `tasks/task_e/metrics.py`：overall、horizon-wise 和 channel-wise 指标

## 1. 任务定义

给定一条 trajectory 过去的 `context_len` 步：

\[
X_{\mathrm{context}}\in\mathbb{R}^{L\times C},
\]

模型预测未来 `pred_len` 步：

\[
\hat X_{\mathrm{future}}\in\mathbb{R}^{H\times C}.
\]

默认配置为：

```text
context_len = 200
pred_len    = 50
stride      = 20
```

对于 `orbital15`，模型输入和输出 shape 分别为：

```text
input  [B, 200, 15]
output [B,  50, 15]
```

模型一次性直接输出完整未来窗口，不进行逐步 autoregressive rollout。

## 2. 数据来源

默认读取：

```text
dataset_v3/results_batch_*_sampling_2.hdf5
```

数据模块会：

1. 根据 glob 找到所有 HDF5 shards；
2. 检查各 shard 的 trajectory 数量、时间长度和 feature shape；
3. 建立全局 trajectory ID 到 `(shard, local_row)` 的映射；
4. 在训练时按需读取窗口，不一次性把全部 trajectory 加载到内存。

## 3. 输入表示

Task E 支持：

```text
orbital15
trajectory
raw_state18
```

其中 `raw_state18` 是 `trajectory` 的别名。

### 3.1 orbital15

通道组成如下：

| Feature | Raw dimension | Transform | Output dimension |
|---|---:|---|---:|
| `aratios` | 1 | `log1p` | 1 |
| `mratios` | 2 | `log1p` | 2 |
| `eccs` | 2 | identity | 2 |
| `incs` | 2 | identity | 2 |
| `APs` | 2 | sin/cos | 4 |
| `LANs` | 2 | sin/cos | 4 |
| Total |  |  | 15 |

因此当前代码中的 `orbital15` 是：

\[
[\log(1+a_r),\log(1+m_r),e,i,
\sin\omega,\cos\omega,\sin\Omega,\cos\Omega].
\]

这里是按 feature group 简写；`mratios`、`eccs`、`incs`、`APs` 和 `LANs` 实际各自包含两个原始分量。

#### 为什么对 ratio 使用 log1p

`aratios` 和 `mratios` 是正值比例变量，可能具有偏斜分布，并且不同样本可能跨越多个数量级。直接输入原值时，大值会对均值、方差和 MSE 产生更强影响。

`log1p` 定义为：

\[
z=\log(1+x).
\]

它有三个主要作用：

1. **压缩大值**：例如 `1 → 0.693`、`10 → 2.398`、`100 → 4.615`，降低极大 ratio 对训练的支配；
2. **减小右偏和异方差**：让长尾正值分布更接近模型容易处理的尺度；
3. **保留零点并避免 `log(0)`**：`log1p(0)=0`，比直接使用 `log(x)` 更稳定。

在 log space 中，模型也更接近于学习相对或乘性变化。例如真实值从 1 变到 2，与从 10 变到 20，都是相近的倍数变化；log 变换会让这两类变化的数值意义更接近。

但 `log1p` 不是无条件正确：

- 必须保证 `x > -1`；
- `-1` sentinel 会产生 `-inf`；
- 如果 ratio 的实际范围很窄，log 变换可能没有明显收益；
- Panda checkpoint 若不是用相同变换预训练，zero-shot 输入分布会发生错配。

因此应该做 `raw ratio` 与 `log1p ratio` 消融，并核对 Panda 预训练数据的确切预处理。建议将两种表示明确命名为 `orbital15_raw` 和 `orbital15_logratio`，避免都叫 `orbital15`。

### 3.2 trajectory/raw_state18

该表示直接拼接三个天体的位置和速度：

```text
positions   3 bodies × 3 axes = 9
velocities  3 bodies × 3 axes = 9
total                           = 18
```

当前统一 pipeline 不会自动将它转换到质心系，也不会消除整体旋转或构造相对坐标。因此它主要是 raw-state baseline。

## 4. 数据切分

当前按 32-variation base system 分组后随机切分：

```text
train = 70%
val   = 15%
test  = 15%
```

系统编号定义为：

```text
system_id = trajectory_id // 32
```

先切分 `system_id`，再把该 system 的全部 variations 分配到同一个 split。即使使用 `--max_trajectories` 限量，同一 system 也不会跨 split。

默认还读取 `dataset_v3/completed_systems_sampling_2.txt`，只保留 diagnosis 有效的完整 32-variation systems。可通过 `--no-require_complete_systems` 放宽为有效的单条 trajectories。

## 5. Scaler 是什么

不同通道的数值范围可能非常不同。例如某些 angle sin/cos 位于 `[-1,1]`，另一些物理量可能有更大的方差。如果直接计算多通道 MSE，高方差通道会主导 loss。

Task E 为每个 feature channel 计算训练集均值和标准差：

\[
\mu_c=\frac{1}{N_c}\sum_n x_{n,c},
\qquad
\sigma_c=\sqrt{\frac{1}{N_c}\sum_n(x_{n,c}-\mu_c)^2}.
\]

“拟合 scaler”就是从数据中估计这些 `mean` 和 `std`。随后将输入和 target 标准化：

\[
z_{n,c}=\frac{x_{n,c}-\mu_c}{\sigma_c}.
\]

例如某一通道在训练集中的均值为 10、标准差为 2：

```text
raw value 10 → z = 0
raw value 12 → z = 1
raw value  8 → z = -1
```

这样各通道通常处于相近尺度，优化器更容易训练，`mse_z` 也更接近“每个标准差单位上的平均误差”。

### 为什么只能用 train trajectories 拟合

validation 和 test 模拟未见数据。如果使用它们计算 `mean/std`，训练 pipeline 就提前获取了测试集分布信息，形成 preprocessing leakage。

正确流程是：

```text
train trajectories → estimate mean/std
                            ↓
          transform train / val / test with the same mean/std
```

不能分别为 train、val、test 拟合不同 scaler，否则模型看到的数值坐标系不同，指标也不再可比。

Scaler 保存为：

```text
scaler.npz
```

其中包含：

- 每个 channel 的 mean；
- 每个 channel 的 standard deviation；
- channel names。

恢复模型进行预测时，必须同时加载训练时的 scaler，不能在新数据上重新拟合一个 scaler。

### 当前 scaler 的一个语义细节

当前 scaler 只使用 completion event 之前、实际可进入训练集的有限训练窗口，不再扫描 trajectory 的失稳后区域。`--max_scaler_trajectories` 可以限制参与统计的训练 trajectories 数量。

## 6. 滑动窗口

每条 trajectory 的窗口起点为：

```text
0, stride, 2 × stride, ...
```

窗口总长度为：

\[
L_{\mathrm{total}}=L_{\mathrm{context}}+L_{\mathrm{pred}}.
\]

然后拆成：

```text
window[:context_len] → input
window[context_len:] → target
```

可以使用以下参数控制数据量：

```text
--max_trajectories
--max_windows_per_trajectory
--max_train_windows
--max_val_windows
--max_test_windows
--window_sampling random|sequential
```

如果限制每条 trajectory 的窗口数量，`sequential` 会保留最早窗口，`random` 会随机选择。正式实验更适合使用 random 或 early/middle/late 分层抽样。

## 7. 非有限值处理

当前参数为：

```text
--drop_nan_windows
--no-drop_nan_windows
```

默认启用 `--drop_nan_windows`。completion table 中的 `time_unbound` 会先映射到 HDF5 时间轴，窗口必须完整结束于该 event 之前。若 completion-filtered 窗口仍包含 NaN/Inf，pipeline 会明确报出 trajectory ID 和 start，不再静默填零；只有显式使用 `--no-drop_nan_windows` 时才会执行有限值替换。

正式实验建议：

这避免了失稳后异常 eccentricity 污染 scaler。未来如需把失稳后的状态也作为预测目标，应使用显式 validity/event mask，而不是将其当作普通连续值。

## 8. 模型

### 8.1 Baselines

- `persistence`：将 context 最后一步复制到所有未来步；不训练；
- `ridge`：flatten 整个 context，直接线性回归完整未来窗口。

Ridge 默认限制为 5000 个训练窗口、2000 个 validation 窗口和 2000 个 test 窗口，因为它需要将窗口物化进内存。

### 8.2 Supervised scratch models

- `tcn`：causal dilated temporal convolution；
- `patchtst`：项目内从零训练、channel-independent 的 PatchTST；
- `transformer`：时间点作为 token 的 Transformer encoder；
- `itransformer` / `itransformer_simple`：变量作为 token 的简化 iTransformer。

当前 `patchtst` 不是 pretrained Hugging Face PatchTST。

### 8.3 Pretrained zero-shot models

- `timesfm`：默认 `google/timesfm-2.5-200m-pytorch`；逐 channel 单变量预测；
- `chronos`：默认 `amazon/chronos-bolt-small`；逐 channel 单变量预测；
- `panda_zero_shot`：默认 `GilpinLab/panda-72M`；直接处理 multivariate `orbital15`。

Panda 源码默认位于：

```text
experiments/panda_threebody/vendor/panda/
```

Panda、TimesFM 和 Chronos 的 pretrained 权重由 Hugging Face cache 管理。

### 8.4 Pretrained adapters

- `timesfm_adapter`：TimesFM inference bridge 后训练 post-adapter；
- `chronos_adapter`：默认冻结 Chronos 和 pre-adapter，训练 post-adapter；
- `panda_adapter`：Panda 前后加入 temporal adapters，默认冻结 Panda backbone。

TimesFM 的公开 forecast 路径会 detach 到 NumPy，因此梯度不能穿过 backbone；当前 TimesFM adapter 更准确地说是对固定 forecast 的 supervised calibration。

## 9. 训练流程

Supervised 和 adapter 模型共享以下流程：

```text
train DataLoader
    ↓
forward
    ↓
MSE loss in z-space
    ↓
backpropagation
    ↓
gradient clipping
    ↓
validation inference
    ↓
select by val_mse_z
    ↓
save best_model.pt
```

默认使用 AdamW、early stopping、gradient clipping，并可选 CUDA mixed precision。

训练目标为：

\[
\mathcal L=\frac{1}{BHC}\sum_{b,h,c}
(\hat z_{b,h,c}-z_{b,h,c})^2.
\]

## 10. Checkpoint

神经模型根据最低 `val_mse_z` 保存：

```text
best_model.pt
```

内容包括：

```python
{
    "epoch": epoch,
    "val_mse_z": best_val,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
}
```

Ridge 保存为 `best_model.pkl`；Persistence 和 zero-shot models 没有训练 checkpoint。

恢复一个神经模型通常需要同时保留：

```text
best_model.pt
config.json
feature_metadata.json
scaler.npz
```

Adapter 还需要对应的原始 pretrained checkpoint。

## 11. Evaluation

当前报告：

- z-space MSE、MAE、RMSE；
- transformed-feature-space MSE、MAE、RMSE；
- z-space 和 transformed-feature-space sMAPE；
- z-space 和 transformed-feature-space Spearman correlation；
- horizon-wise metrics；
- channel-wise metrics；
- KL divergence；
- Hellinger distance。

inverse scaler 只撤销 z-score，没有撤销 `log1p` 或 sin/cos，因此该空间是 transformed feature space，而不是真正的原始物理变量空间。新输出使用 `*_transformed`；为了兼容已有结果读取，暂时同时保留同值的 legacy `*_raw` 字段。

sMAPE 以百分数保存为 `smape_percent_z` 和 `smape_percent_raw`，范围为
`[0, 200]`。Spearman 使用所有有限 prediction/target pairs 的秩相关，保存为
`spearman_z` 和 `spearman_raw`；常数序列由于秩相关未定义而返回 NaN。

真正的 physical metrics 需要：

- ratio 使用 `expm1`；
- angle 使用 `atan2(sin, cos)`；
- 使用 circular angular error；
- 检查预测 sin/cos 的单位圆误差。

## 12. 实验产物

每次运行必须指定 `--output_dir`：

```bash
python -m tasks.task_e \
  --model tcn \
  --representation orbital15 \
  --context_len 200 \
  --pred_len 50 \
  --output_dir runs/task_e_tcn_ctx200_pred50
```

典型输出为：

```text
runs/task_e_tcn_ctx200_pred50/
├── config.json
├── feature_metadata.json
├── scaler.npz
├── train_log.csv
├── best_model.pt
├── val_metrics.json
├── test_metrics.json
├── horizon_metrics.csv
├── channel_metrics.csv
└── predictions_sample.npz
```

Zero-shot 模型不会生成 `best_model.pt` 和 `train_log.csv`。

## 13. 当前优点

- baseline、scratch、pretrained 和 adapter 共用统一入口；
- scaler 只使用有效、事件前的 train windows；
- 同一个 32-variation base system 不跨 split；
- HDF5 lazy loading；
- 保存完整 config、scaler 和 feature metadata；
- validation 选择 checkpoint；
- 支持 horizon-wise 和 channel-wise 评价；
- foundation dependencies 缺失时不影响本地 supervised models。

## 14. 正式实验前的优先修改项

1. 核对 Panda checkpoint 与 `orbital15 + log1p ratios` 的预处理一致性；
2. 实现完整 physical inverse transform；
3. validation/test 指标改为流式累计，避免保存全量 context/prediction；
4. 增加 circular angle、sin/cos norm、物理有效性和守恒指标；
5. 固定并导出所有模型共用的 test window IDs；
6. 对 `raw ratio` 与 `log1p ratio` 做消融实验。
