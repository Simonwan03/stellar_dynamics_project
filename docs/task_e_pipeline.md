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
- `tasks/task_e/foundation.py`：Chronos-2 和 Panda 加载与 zero-shot 推理
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
context_len = 512
pred_len    = 128
stride      = 20
```

对于 `orbital15`，模型输入和输出 shape 分别为：

```text
input  [B, 512, 15]
output [B, 128, 15]
```

模型一次性直接输出完整未来窗口，不进行逐步 autoregressive rollout。

Task E 现在包含两个共享同一数据切分、窗口和 train-only scaler 的子实验：

1. **E1 one-shot forecasting**：每个模型直接完成 `512 -> 128` 预测，研究 immediate future trajectory 的预测准确度。
2. **E2 rolling forecasting**：复用同一个 E1 模型，连续进行四个 `128` 步 recursive blocks，得到最长 `512` 步预测；后续 block 只使用模型自己的预测更新 context，不使用未来 ground truth，研究误差累积速度。

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

### 2.1 独立预处理缓存

全量 HDF5 的 finite 检查和 train-only scaler 计算可以单独执行一次。预处理命令只保存窗口索引、有效长度、scaler、feature metadata 和输入文件签名，不复制 HDF5 数据：

```bash
python -m tasks.task_e.prepare \
  --cache-dir runs/task_e_prepared_cache_orbital15_ctx512_pred128 \
  --data-glob "dataset_v3/results_batch_*_sampling_2.hdf5" \
  --completion-table dataset_v3/completed_systems_sampling_2.txt \
  --representation orbital15 \
  --context-len 512 --pred-len 128 --stride 20 \
  --drop-nan-windows
```

之后每个模型运行时传入同一个缓存目录：

```bash
python -m tasks.task_e \
  --prepared-cache runs/task_e_prepared_cache_orbital15_ctx512_pred128 \
  --model chronos2 \
  --foundation_batch_size 32 \
  --output-dir runs/task_e_chronos2
```

缓存加载会校验 HDF5 文件大小/修改时间、completion table 和所有影响窗口/scaler 的设置；配置不一致时会直接报错，避免误用旧缓存。E2 rolling 的少量长窗口仍会在启用 `--run-rolling` 时做自身的完整性检查。

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
--max_train_variations_per_system
--max_windows_per_trajectory
--max_train_windows
--max_val_windows
--max_test_windows
--window_sampling random|sequential
--train-windows-per-trajectory 8
--val-windows-per-trajectory 4
--test-windows-per-trajectory 4
--rolling-windows-per-trajectory 2
--bootstrap-samples 1000
```

`stride=20` 只用于生成 candidate window starts，不意味着所有高度重叠窗口都必须在
每个 epoch 中训练或评估。candidate windows 会先保留完整；默认 sampling protocol 为：

```text
train    每个 epoch 每条 trajectory 随机采样最多 8 个窗口
val      每条 trajectory 固定采样最多 4 个窗口，位置约为 10/35/65/90%
E1 test  每条 trajectory 固定采样最多 4 个窗口，位置约为 10/35/65/90%
E2 test  每条 trajectory 固定采样最多 2 个 rolling starts，位置约为 25/75%
```

training sampling 每个 epoch 使用新的 deterministic random state，因此同一条
trajectory 会在不同 epoch 覆盖不同阶段；validation/test sampling 不随 epoch 变化，
所有模型使用完全相同的 evaluation windows。`--train-windows-per-trajectory 0`、
`--val-windows-per-trajectory 0`、`--test-windows-per-trajectory 0` 和
`--rolling-windows-per-trajectory 0` 可恢复使用全部 eligible candidate windows。

当前实现优先保证 trajectory-balanced sampling：每条 eligible trajectory 每个 epoch
最多贡献固定数量窗口。由于 trajectory 属于唯一的 base system，这已经消除了长寿命
trajectory 在 window-level training 中的主要过度加权；最终 evaluation 仍进一步按
system 等权聚合。

`--max_train_variations_per_system` 在完成 train/validation/test 的 system-level
切分之后，仅对训练集每个 system 内的 variations 做随机上限。例如：

```bash
--max_train_variations_per_system 1
```

表示每个训练用的 32-variation system 最多保留一条 trajectory。它不会把同一个
system 拆到不同 split，也不会减少 validation/test 中的 variations。默认值 `0`
表示训练集保留所有 variations。若需要保持每个 system 的代表性，建议配合固定
`--split_seed` 和 `--seed` 使用。

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

- `ridge`：flatten 整个 context，直接线性回归完整未来窗口。

Ridge 默认限制为 5000 个训练窗口、2000 个 validation 窗口和 2000 个 test 窗口，因为它需要将窗口物化进内存。

### 8.2 Supervised scratch models

- `tcn`：causal dilated temporal convolution；
- `patchtst`：项目内从零训练、channel-independent 的 PatchTST；
- `transformer`：时间点作为 token 的 Transformer encoder；
- `itransformer`：变量作为 token 的简化 iTransformer。

当前 `patchtst` 不是 pretrained Hugging Face PatchTST。

### 8.3 Pretrained zero-shot models

- `chronos2`：默认 `amazon/chronos-2`；使用原生 multivariate pipeline，输入
  `[B,T,C]` 会转换为 Chronos-2 所需的 `[B,C,T]`，并取中位 quantile 作为点预测；
- `panda_zero_shot`：默认 `GilpinLab/panda-72M`；直接处理 multivariate
  `orbital15` 或 `raw_state18`。Panda 的共享通道 embedding/forecast head
  支持运行时通道数，因此 15 通道和 18 通道都可传入。

Chronos-2 示例：

```bash
conda run -n stellar python -m tasks.task_e \
  --model chronos2 \
  --chronos_model_id amazon/chronos-2 \
  --context_len 512 \
  --pred_len 128 \
  --foundation_batch_size 16 \
  --output_dir runs/task_e_chronos2
```

Chronos-2 的 `prediction_length` 超过 checkpoint 默认 horizon 时，官方 API
可能发出质量提醒并使用长 horizon 策略。

Panda 源码默认位于：

```text
experiments/panda_threebody/vendor/panda/
```

Panda 和 Chronos-2 的 pretrained 权重由 Hugging Face cache 管理。

### 8.4 Panda zero-shot

`panda_zero_shot` 直接使用 Panda 的原生 `512 -> 128` 配置，不再经过
`panda_adapter`。`panda_adapter` 仍保留少量历史代码以便读取旧实验，但不再是当前
Task E 的公开模型选项，也不会参与 E1/E2 benchmark。

### 8.5 E1 one-shot 与 E2 rolling evaluation

E1 的代表性模型可以分别运行：

```bash
conda run -n stellar python -m tasks.task_e \
  --data_glob "dataset_v3/results_batch_*_sampling_2.hdf5" \
  --representation orbital15 \
  --model panda_zero_shot \
  --context-len 512 \
  --pred-len 128 \
  --output-dir runs/task_e_e1_panda_zero_shot
```

当前公开的代表性模型包括 `ridge`、`tcn`、`patchtst`、`itransformer`、
`chronos2` 和 `panda_zero_shot`。所有模型共享相同的 candidate pool、system split、
deterministic validation/test windows 和 train-only scaler。

在同一次运行中追加 `--run-rolling` 即可执行 E2，不会重新训练模型：

```bash
conda run -n stellar python -m tasks.task_e \
  --model panda_zero_shot \
  --representation orbital15 \
  --context-len 512 \
  --pred-len 128 \
  --rolling-blocks 4 \
  --run-rolling \
  --output-dir runs/task_e_e1e2_panda_zero_shot
```

只有真实轨迹仍有完整 `512` 步 future 且全部有限的 rolling starts 才进入 E2。
E2 默认从每条 trajectory 的 rolling candidate starts 中固定选择最多两个分散位置，
并保存 `rolling_horizon_mae`、四个 block MAE 和
`error_amplification_ratio = block_4_mae / block_1_mae`。


## 9. 训练流程

Supervised 模型共享以下流程：

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

Task E 的统计层级为：

```text
system
  -> trajectory / variation
      -> forecasting window
```

system split、bootstrap 和最终正文 summary 都以 system 为统计单位。任意 scalar
error 先在 window 内计算，再按 trajectory 求均值、按 system 对 trajectory 均值，
最后对 system 等权求均值。原来的 pooled window-weighted 值仍保存为 diagnostic，
例如 `mae_z_window_weighted`；正文使用 `mae_z_system_balanced`，并将其兼容别名
保存为 `mae_z`。

E1 的正文核心指标为：

- `mse_z`、`mae_z`、`rmse_z`；
- Pearson correlation；
- `horizon_mae[1:128]`；
- `unit_circle_error`：对 orbital15 的 AP/LAN 两个角度各自的 sin/cos pair 计算
  `|sin² + cos² - 1|`，并报告四个 pair 的总体均值。
- `horizon_mae_system_balanced[1:128]`，兼容别名为 `horizon_mae`。
- `bootstrap_ci95`：对 `mse_z`、`mae_z`、`rmse_z` 做 system-level bootstrap 95% CI，
  默认 `--bootstrap-samples 1000`。

同时保留以下 secondary metrics：

- z-space MSE、MAE、RMSE；
- transformed-feature-space MSE、MAE、RMSE；
- Pearson/Spearman correlation；
- z-space 和 transformed-feature-space sMAPE；
- horizon-wise metrics；
- channel-wise metrics；
- KL divergence；
- Hellinger distance。

E2 另外报告：

- `rolling_horizon_mae[1:512]`；
- `rolling_horizon_mae_system_balanced[1:512]`；
- `block_1_mae`、`block_2_mae`、`block_3_mae`、`block_4_mae`；
- `error_amplification_ratio`。
- 四个 block MAE 和 amplification ratio 的 system-level bootstrap 95% CI。

每个模型的运行目录保存 `e1_metrics.json`、`rolling_metrics.json`（启用 E2 时）、
`summary.json`、`summary.csv` 和 `summary.md`。使用
`python -m tasks.task_e.report --run-dirs ... --output-dir ...` 可汇总多个模型并生成
`one_shot_horizon_mae.png` 与 `rolling_horizon_mae.png`。

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

每次运行必须指定 `--output_dir`。`config.json` 的 `sampling_metadata` 会同时保存：

```text
candidate_*_windows
sampled_train_windows_per_epoch
sampled_val_windows
sampled_e1_test_windows
sampled_e2_rolling_windows
num_*_systems
num_*_trajectories
*_windows_per_trajectory
```

这样可以明确区分 candidate pool 和实际训练/评估的窗口数量。

每次运行必须指定 `--output_dir`：

```bash
python -m tasks.task_e \
  --model tcn \
  --representation orbital15 \
  --context_len 512 \
  --pred_len 128 \
  --output_dir runs/task_e_tcn_ctx512_pred128
```

典型输出为：

```text
runs/task_e_tcn_ctx512_pred128/
├── config.json
├── feature_metadata.json
├── scaler.npz
├── train_log.csv
├── best_model.pt
├── val_metrics.json
├── test_metrics.json
├── horizon_metrics.csv
├── channel_metrics.csv
├── e1_metrics.json
├── summary.json
├── summary.csv
├── summary.md
└── predictions_sample.npz
```

启用 `--run-rolling` 后还会生成：

```text
├── rolling_metrics.json
├── e2_metrics.json
└── rolling_horizon_metrics.csv
```

Zero-shot 模型不会生成 `best_model.pt` 和 `train_log.csv`。E1/E2 的多模型图表由
`tasks.task_e.report` 根据各模型运行目录汇总生成。

## 13. 当前优点

- baseline、scratch、pretrained 和 adapter 共用统一入口；
- scaler 只使用有效、事件前的 train windows；
- 同一个 32-variation base system 不跨 split；
- HDF5 lazy loading；
- 保存完整 config、scaler 和 feature metadata；
- validation 使用固定 deterministic windows 并按 system-balanced `mse_z` 选择 checkpoint；
- 支持 system-balanced scalar、horizon-wise、channel-wise 评价和 system bootstrap CI；
- foundation dependencies 缺失时不影响本地 supervised models。

## 14. 正式实验前的优先修改项

1. 核对 Panda checkpoint 与 `orbital15 + log1p ratios` 的预处理一致性；
2. 实现完整 physical inverse transform；
3. 增加 circular angle、sin/cos norm、物理有效性和守恒指标；
4. 对 `raw ratio` 与 `log1p ratio` 做消融实验。
