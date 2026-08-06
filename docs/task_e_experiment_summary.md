# Task E 实验结果汇总

生成日期：2026-07-13。本文根据 `runs/**/config.json`、`val_metrics.json`、`test_metrics.json` 与 `train_log.csv` 汇总当前工作区中保存的所有 Task E 相关运行；数值直接来自对应运行目录，并非重新评估。

> 注：Persistence 运行目录和结果表作为历史记录保留；当前统一入口已不再提供该模型。

## 口径与可比性

- 统一 Task E 基准使用 `orbital15`（15 通道）、输入 200 步、预测 50 步、`split_seed=42`、按 32 个变体组成的系统分组切分，并过滤未完成系统。
- 表中 `MSE`、`MAE` 与 `Spearman` 均为**原始尺度**（`*_raw`）测试指标；MSE/MAE 越低越好，Spearman 越高越好。
- “完全可比”指 512 条轨迹、20 个数据文件、顺序采样，以及相同的 292 个验证窗口和 305 个测试窗口。`panda_adapter_full_eval_20260712` 已在此完整评估集上运行；其训练窗口数仍少于其他监督模型。`panda_zero_shot` 的非 comparable 版本和早期 Panda Adapter 探索运行使用了不同的评估窗口数量，须单独解读。
- `panda_temporal_adapter_full*` 是历史的独立 Panda 管线：窗口构造、预测长度（256→64）与度量实现均不同，不能和统一 Task E 表横向比较。

## 统一基准：完全可比的结果

共同设置：512 条轨迹，训练/验证/测试窗口数为 1398 / 292 / 305，除非表中另有说明。

| 模型（运行目录） | 训练方式 | 训练 epoch | 验证 MSE | 测试 MSE | 测试 MAE | 测试 Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Panda Adapter（完整评估，`.../v2/panda_adapter_full_eval_20260712`）† | 冻结主干的适配器监督训练 | 10 | **0.098100** | **0.094817** | **0.161361** | **0.897936** |
| TimesFM zero-shot (`.../v2/timesfm`) | 零样本 | — | 0.122229 | 0.115029 | 0.177769 | 0.876202 |
| Panda zero-shot (`.../v2/panda_zero_shot_comparable`) | 零样本 | — | 0.136210 | 0.124031 | 0.198511 | 0.860429 |
| Chronos T5-small (`.../v2/chronos_t5_small`) | 零样本 | — | 0.153873 | 0.138688 | **0.165562** | 0.865887 |
| PatchTST (`.../v2/patchtst`) | 全监督 | 4 | 0.189831 | 0.172052 | 0.252842 | 0.797724 |
| iTransformer (`.../v2/itransformer`) | 全监督 | 4 | 0.193811 | 0.177432 | 0.266774 | 0.781257 |
| TCN (`.../compact_20260712/tcn`) | 全监督 | 10 | 0.216249 | 0.195304 | 0.292288 | 0.761872 |
| Persistence (`.../compact_20260712/persistence`) | 最后值基线 | — | 0.355868 | 0.325589 | 0.309381 | 0.689615 |
| Ridge (`.../v2/ridge`) | 线性基线 | — | 0.998017 | 0.949673 | 0.583040 | 0.537908 |

† Panda Adapter 的**评估**与本表一致（292/305 窗口），但其适配器训练只使用 256 个窗口；其他监督模型使用 1398 个训练窗口。因此表中的测试指标可直接比较，训练数据预算则不同。

结论：在当前完整统一评估集上，Panda Adapter 的测试 MSE（0.094817）、MAE（0.161361）和 Spearman（0.897936）均为已保存运行中的最佳值。TimesFM 是最强零样本基线（MSE 0.115029），Chronos T5-small 的 MAE 为 0.165562。保留的 10-epoch TCN 结果为测试 MSE 0.195304、MAE 0.292288、Spearman 0.761872，仍落后于基础模型与 Panda Adapter。

## raw18 / raw_state18：位置与速度特征结果

该批次使用 18 通道原始状态（3 个天体的 xyz 位置和 xyz 速度），其余设置与主基准一致：512 条轨迹、20 个数据文件、200→50、`split_seed=42`、32 变体分组切分、顺序采样与完成系统过滤。得到 1400 / 292 / 306 个训练/验证/测试窗口。由于 raw18 和 orbital15 的物理量及变换尺度不同，本表仅报告并用于 **raw18 内部比较** 的标准化测试指标（`*_z`）；不可与上方 orbital15 的 `*_raw` 指标直接横向比较。

| 模型（运行目录） | 训练方式 | 训练 epoch | 测试 MSE_z | 测试 MAE_z | 测试 Spearman_z |
| --- | --- | ---: | ---: | ---: | ---: |
| TimesFM zero-shot (`.../raw18.../timesfm`) | 零样本 | — | **0.525152** | **0.349141** | 0.424185 |
| Chronos T5-small (`.../raw18.../chronos`) | 零样本 | — | 0.580466 | 0.355880 | **0.428264** |
| TCN (`.../raw18.../tcn`) | 全监督 | 10 | 0.623281 | 0.431166 | 0.220818 |
| iTransformer (`.../raw18.../itransformer`) | 全监督 | 4 | 0.623715 | 0.419106 | 0.193764 |
| PatchTST (`.../raw18.../patchtst`) | 全监督 | 4 | 0.626541 | 0.419600 | 0.197222 |
| Transformer (`.../raw18.../transformer`) | 全监督 | 4 | 0.906040 | 0.481039 | 0.155406 |
| Persistence (`.../raw18.../persistence`) | 最后值基线 | — | 1.054121 | 0.525768 | 0.204258 |
| Ridge (`.../raw18.../ridge`) | 线性基线 | — | 2.595065 | 0.965088 | 0.099415 |

该批次运行时 Task E 入口仍人为限制 Panda 只能使用 `orbital15`，所以没有保存
raw18 的 Panda zero-shot 或 Panda Adapter 指标；这不是 Panda 模型本身的通道
限制。当前入口已允许 Panda 直接接收 18 通道 `raw_state18`，但需要另行补跑，
不能把新结果追填为本批次的历史结果。Chronos 因 18 通道展开推理导致原批次大小
32 显存不足，改用 `foundation_batch_size=4` 完成；这只改变推理分批方式，不改变
预测样本或指标。

## Panda Adapter：训练与早期小窗口评估

共同训练设置：512 条轨迹、256 个训练窗口、冻结 `GilpinLab/panda-72M` 主干、`adapter_lr=1e-4`、batch size 1。下表是最初仅在 128/128 验证/测试窗口上得到的探索性评估；完整评估结果已列入上方主表。

| 运行目录 | 检查点/训练过程 | 最佳验证 epoch | 验证 MSE | 测试 MSE | 测试 MAE | 测试 Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `runs/task_e_prelim_compact_20260712_v2/panda_adapter` | 初始训练，epoch 1–6 | 4 | 0.100773 | **0.098633** | 0.169172 | 0.891364 |
| `runs/task_e_prelim_compact_20260712_v2/panda_adapter_resume_20260712` | 从初始最佳 checkpoint 恢复，额外训练 epoch 5–10 | 8 | **0.098271** | 0.098756 | **0.168499** | **0.891965** |

续训在小验证集上有所改善（原始 MSE 0.100773 → 0.098271），并在完整验证/测试集上重新评估：292 窗口验证 MSE 为 0.098100，305 窗口测试 MSE 为 0.094817。续训日志显示小验证集在 epoch 8 后回升；完整评估加载的正是 epoch 8 的最佳 checkpoint。

## 其他统一 Task E 运行

| 运行目录 | 数据规模与采样 | 已保存的结果 | 说明 |
| --- | --- | --- | --- |
| `runs/task_e_prelim_compact_20260712_v2/panda_zero_shot` | 512 条轨迹；验证/测试各 256 窗口 | 测试 MSE 0.122947，MAE 0.198808，Spearman 0.861434 | 与 comparable Panda 版本仅评估窗口数不同，数值接近；不作为完整 305 窗口排行榜的条目。 |
| `runs/task_e_prelim_small_20260712_v2/persistence` | 2048 条轨迹；顺序采样；10946/2000/2000 窗口 | 测试 MSE 0.336771，MAE 0.317444，Spearman 0.675598 | 更大的数据子集，仅有持久性基线，不能与 512 条轨迹表直接比较。 |
| `runs/task_e_prelim_small_20260712/persistence` | 2048 条轨迹；随机采样；10948/2000/2000 窗口 | 无 `val_metrics.json` / `test_metrics.json` | 配置和特征元数据已保存，但未保存最终评估。 |
| `runs/task_e_prelim_small_20260712_seq/persistence` | 2048 条轨迹；顺序采样；10948/2000/2000 窗口 | 无 `val_metrics.json` / `test_metrics.json` | 配置和特征元数据已保存，但未保存最终评估。 |

## 历史独立 Panda 管线

| 运行目录 | 数据与设置 | 结果状态 |
| --- | --- | --- |
| `runs/panda_temporal_adapter_full` | `orbital15`，256→64，原始 4096 窗口、过滤后 2980，训练/验证/测试 2086/447/447，10 epoch | 仅保存配置；未保存训练日志或测试指标。 |
| `runs/panda_temporal_adapter_full_20k` | `orbital15`，256→64，原始 20000 窗口、过滤后 14522，训练/验证/测试 10165/2178/2179，冻结 Panda，10 epoch | 测试 `z_mse=0.173401`、`z_mae=0.191221`；该管线的 `raw_mse=1992.260254`、`raw_mae=2.509673`。训练验证损失在第 10 epoch 最低（`val_mse=1788.239416`）。 |

## 建议的后续比较

1. 对 Panda Adapter 使用独立验证集选择 checkpoint（当前为续训 epoch 8）；今后的测试实验应避免在同一测试集上反复挑选模型。此前的 128 窗口测试结果应视为探索性记录。
2. 将 Panda Adapter 的训练窗口数由 256 扩展至 1398，量化训练数据预算对其领先结果的影响。
3. 扩展全监督模型的训练预算并记录早停状态。现有 PatchTST、iTransformer 和 TCN 仅训练 4 或 10 epoch，尚不足以判断它们的充分训练上限。
4. 历史 persistence 运行不再纳入当前模型比较；未完成目录仅作历史记录，若需清理应单独处理。

## 结果位置

- 统一基准配置和指标：`runs/task_e_prelim_compact_20260712*/**/{config,val_metrics,test_metrics}.json`
- Panda Adapter 续训：`runs/task_e_prelim_compact_20260712_v2/panda_adapter_resume_20260712/`
- Panda Adapter 完整评估：`runs/task_e_prelim_compact_20260712_v2/panda_adapter_full_eval_20260712/`
- raw18 批次：`runs/task_e_raw18_matched_20260713/`
- 历史 Panda 管线：`runs/panda_temporal_adapter_full*`
