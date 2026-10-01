# Historical Task D experiments

These are preserved research records. They use earlier quantile or sampling 2 targets and do not reproduce the current Task D median-decorrelation result. The text below preserves the old terminology and commands for provenance; its references to a "current" task describe the earlier code, and several flags are unsupported by the present CLI. Current definitions and commands are in [README](../readme.md#task-d-median-orbital-decorrelation-time) and [reproduction map](../docs/reproduction.md).

### Task D: Initial Conditions → Lyapunov-Time Quantiles

入口脚本：[tasks/task_d/main.py](../tasks/task_d/main.py)

底层实现：[tasks/task_d/quantile.py](../tasks/task_d/quantile.py)

Task D 现在只使用 system 的初始条件，预测 system-level semi-major-axis-ratio
Lyapunov-time proxy 的条件 P25/P50/P75。对每个 `(i, i+16)` pair 定义

`G_ij(t)=|Q_i(t)-Q_j(t)|/|Q_i(0)-Q_j(0)|`, `Q=a_in/a_out`。

默认 pair label 为 `T_ij=t_cross(G=e^5)/5`。system label 先对有效 pair 的
`lambda_ij=1/T_ij` 取中位数，再取倒数；至少要求8个有效 pair。模型训练目标是
`log(T_system/P_out,0)`。它是基于指定 observable 的 finite-time proxy，不是变分方程
定义的最大 Lyapunov time。

输入由前16个 nominal variations 的 `orbital15` 初始条件组成，角度使用 sin/cos。
GBR/MLP 使用 permutation-invariant 的 mean/std/min/max 汇总，DeepSets 直接编码16元素
集合。三个输出使用 pinball loss，并通过有序输出头保证 `q25 <= q50 <= q75`。

```bash
conda run -n stellar python -m tasks.task_d \
  --dataset-dir dataset_sampling_3_prefix1024 \
  --threshold-power 5 \
  --models constant,gbr,mlp,deepsets \
  --output-dir results/task_d/sampling_3/task_d_initial_quantiles \
  --device cuda
```

默认按 system 做 70/15/15 train/validation/test split。评价包括各分位数 pinball
loss 和 empirical coverage、中心50%区间 coverage/width、P50 MAE/RMSE/Spearman、
factor-2 accuracy 与 quantile crossing rate。

推荐同时运行 `e^4/e^5/e^6` label 敏感性实验：

```bash
conda run -n stellar python -m tasks.task_d.sensitivity \
  --threshold-powers 4 5 6 \
  --models constant,gbr \
  --output-dir results/task_d/sampling_3/task_d_initial_quantile_sensitivity
```

#### Label-threshold sensitivity ablation (2026-09-08)

在相同 seed=42、特征表示和模型配置下，比较 crossing threshold 为 `e^4`、`e^5`、
`e^6` 的结果。下表报告 GBR 的主要测试集指标；完整的 constant baseline、逐分位数
coverage 和预测文件见 [`results/task_d/sampling_3/task_d_initial_quantile_sensitivity/summary.md`](../results/task_d/sampling_3/task_d_initial_quantile_sensitivity/summary.md)。

| Threshold | 有效 systems | Target P25/P50/P75 | GBR mean pinball ↓ | GBR P50 log-MAE ↓ | Central 50% coverage | Spearman ↑ | Factor-2 accuracy ↑ |
|---:|---:|---|---:|---:|---:|---:|---:|
| `e^4` | 4,997 | 0.9544 / 3.4319 / 9.1262 | **0.3227** | **0.7559** | 0.4827 | 0.6589 | **54.13%** |
| `e^5` | 4,996 | 0.9113 / 3.4091 / 11.2786 | 0.3611 | 0.8475 | 0.4780 | 0.6686 | 51.27% |
| `e^6` | 4,988 | 0.8811 / 3.3414 / 12.0871 | 0.3930 | 0.9139 | 0.4920 | **0.6810** | 48.13% |

GBR 在三个 threshold 下都优于 constant quantile baseline。threshold 增大后目标分布的
上尾变宽，GBR 的 pinball loss 和绝对误差逐步上升，而中心 50% 区间 coverage 仍接近
名义值。由于“至少 8 个有效 pair”的过滤，三组有效 system 并非完全相同，因此趋势应
解释为 label sensitivity，而不是严格配对比较。

默认在相邻存储点之间按 log growth 插值 crossing；使用 `--no-interpolation` 可改为
首次离散 crossing。旧的 single-target crossing/slope 实现仍保留在
`tasks/task_d/pipeline.py`，仅用于复核历史结果，不再由 Task D 默认入口调用。

### 4. Task D：Lyapunov-proxy / lambda-fast 回归

Task D 的 proxy 标签由 16 个 pair 的离散 crossing time 聚合得到；数据采样间隔约为
50000，因此这里的时间是“首次存储采样点”而不是连续 crossing time。下面先列出已经
保存的 initial/prefix 结果，再列出 threshold × aggregation 敏感性网格。

#### 4.1 已保存的 initial/prefix 结果

表中的 MAE/RMSE 都是在 log-time 上计算；F2/F3 表示预测落在真实值 2 倍/3 倍范围内
的比例。lambda_fast 与 lyapunov_proxy 是两种不同 target，不能直接横向混排。

| Run | Target / setting | Representation | Labels | Censoring | Model | MAE | RMSE | Spearman | F2 / F3 |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| D-lambda-raw | lambda_fast / initial | raw IC | 3199 | 0.0004 before quality filter | constant median | 0.958787 | 1.757718 | — | 0.6375/0.8094 |
| D-lambda-raw | lambda_fast / initial | raw IC | 3199 | 0.0004 before quality filter | random forest | 1.074405 | 1.622455 | 0.208866 | 0.4688/0.6875 |
| D-lambda-raw | lambda_fast / initial | raw IC | 3199 | 0.0004 before quality filter | MLP | 0.962735 | 1.645267 | 0.235682 | 0.5734/0.7625 |
| D-lambda-orbital15 | lambda_fast / initial | orbital15 | 3199 | 0.0004 before quality filter | constant median | 0.958787 | 1.757718 | — | 0.6375/0.8094 |
| D-lambda-orbital15 | lambda_fast / initial | orbital15 | 3199 | 0.0004 before quality filter | random forest | 1.092392 | 1.654524 | 0.188678 | 0.4469/0.6500 |
| D-lambda-orbital15 | lambda_fast / initial | orbital15 | 3199 | 0.0004 before quality filter | MLP | 1.021682 | 1.679629 | 0.197611 | 0.5266/0.7297 |
| D-proxy-initial | lyapunov_proxy, threshold=1000, min | orbital15 | 4997 | 0.0002 | constant median | 1.000961 | 1.278772 | — | 0.3590/0.7680 |
| D-proxy-initial | lyapunov_proxy, threshold=1000, min | orbital15 | 4997 | 0.0002 | random forest | 0.748865 | 0.984519 | 0.667342 | 0.5630/0.7560 |
| D-proxy-initial | lyapunov_proxy, threshold=1000, min | orbital15 | 4997 | 0.0002 | MLP | **0.652977** | **0.878598** | **0.721417** | **0.6300/0.8200** |
| D-prefix | lambda_fast / initial_prefix, prefix=100 | orbital15 | 4998 | 0.886154 | constant median | 1.642400 | 2.607517 | — | 0.3610/0.5070 |
| D-prefix | lambda_fast / initial_prefix, prefix=100 | orbital15 | 4998 | 0.886154 | random forest | **1.237786** | **2.084188** | **0.657684** | **0.4710/0.6240** |
| D-prefix | lambda_fast / initial_prefix, prefix=100 | orbital15 | 4998 | 0.886154 | MLP | 1.647709 | 3.003794 | 0.495328 | 0.3470/0.5320 |

当前代码推荐的 threshold=10000 版本不是上面 D-proxy-initial 目录中的历史
threshold=1000 结果；推荐使用后面的 sensitivity cell 或重新运行主命令后再作正式比较。

#### 4.2 threshold / aggregation sensitivity

标签分布（每个 cell 都有 4998 个 systems）：

| Threshold | Aggregation | Uncensored | Censored fraction | Median tau | P90 tau |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | median | 4992 | 0.001200 | 300000 | 3957500 |
| 1000 | min | 4997 | 0.000200 | 150000 | 1100000 |
| 3000 | median | 4974 | 0.004802 | 350000 | 5550000 |
| 3000 | min | 4989 | 0.001801 | 150000 | 1300000 |
| 10000 | median | 4921 | 0.015406 | 387500 | 6590000 |
| 10000 | min | 4968 | 0.006002 | 150000 | 1565000 |
| 30000 | median | 4762 | 0.047219 | 425000 | 7615000 |
| 30000 | min | 4934 | 0.012805 | 200000 | 1800000 |

每个 cell 的所有模型指标：

| Threshold | Agg. | Model | MAE(log tau) | RMSE | Spearman | F2 | F3 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1000 | median | constant | 1.2960 | 1.5767 | — | 0.2442 | 0.5385 |
| 1000 | median | MLP | **0.7882** | **1.0205** | **0.7957** | **0.5405** | **0.7558** |
| 1000 | median | RF | 0.9099 | 1.1499 | 0.7005 | 0.4725 | 0.6757 |
| 1000 | min | constant | 1.0010 | 1.2788 | — | 0.3590 | 0.7680 |
| 1000 | min | MLP | **0.6794** | **0.9105** | **0.7158** | **0.6230** | **0.8050** |
| 1000 | min | RF | 0.7489 | 0.9845 | 0.6673 | 0.5630 | 0.7560 |
| 3000 | median | constant | 1.3552 | 1.6973 | — | 0.2673 | 0.4523 |
| 3000 | median | MLP | **0.8769** | **1.1768** | **0.7579** | **0.5206** | **0.6965** |
| 3000 | median | RF | 0.9918 | 1.2911 | 0.6812 | 0.4503 | 0.6332 |
| 3000 | min | constant | 1.0379 | 1.3253 | — | 0.3417 | 0.7365 |
| 3000 | min | MLP | **0.6798** | **0.9164** | **0.7354** | **0.6182** | **0.8116** |
| 3000 | min | RF | 0.7665 | 1.0148 | 0.6527 | 0.5521 | 0.7595 |
| 10000 | median | constant | 1.3625 | 1.7322 | — | 0.2853 | 0.4426 |
| 10000 | median | MLP | **0.8712** | **1.2209** | **0.7621** | **0.5350** | **0.7411** |
| 10000 | median | RF | 1.0037 | 1.3205 | 0.6831 | 0.4508 | 0.6487 |
| 10000 | min | constant | 1.1449 | 1.5105 | — | 0.3239 | 0.6992 |
| 10000 | min | MLP | **0.7297** | **0.9992** | **0.7891** | **0.6137** | **0.7907** |
| 10000 | min | RF | 0.8304 | 1.1433 | 0.6734 | 0.5151 | 0.7334 |
| 30000 | median | constant | 1.3171 | 1.6694 | — | 0.2844 | 0.4533 |
| 30000 | median | MLP | **0.8599** | **1.1857** | **0.7515** | **0.5331** | **0.7272** |
| 30000 | median | RF | 0.9898 | 1.3002 | 0.6574 | 0.4407 | 0.6537 |
| 30000 | min | constant | 1.1276 | 1.4417 | — | 0.2523 | 0.5137 |
| 30000 | min | MLP | **0.7644** | **1.0704** | **0.7383** | **0.5735** | **0.7771** |
| 30000 | min | RF | 0.8504 | 1.1654 | 0.6490 | 0.5360 | 0.7224 |

在当前 sensitivity 网格里，min aggregation 通常保留更多 uncensored labels，MLP 的
log-time MAE 约为 0.679–0.764；median aggregation 的 censoring 更高，MLP MAE
约为 0.788–0.877。完整原始表见
[task_d_proxy_sensitivity directory](../results/task_d/sampling_2/task_d_proxy_sensitivity)。
各 initial/prefix run 的 config、feature importance 和 predictions 分别保存在：
[lambda-fast raw](../results/task_d/sampling_2/task_d_initial_lambda_fast/metrics.json)、
[lambda-fast orbital15](../results/task_d/sampling_2/task_d_initial_lambda_fast_orbital15/metrics.json)、
[lyapunov proxy](../results/task_d/sampling_2/task_d_initial_lyapunov_proxy/metrics.json)、
[prefix](../results/task_d/sampling_2/task_d_initial_prefix_lambda_fast_orbital15/metrics.json)。

#### 4.3 Task D feature importance

已保存 feature importance 中的第一重要特征为：

| Run | Top feature | Importance |
| --- | --- | ---: |
| lambda_fast raw | periods_complete_0 | 0.132797 |
| lambda_fast orbital15 | eccs_complete_1 | 0.143158 |
| lyapunov proxy | incs_complete_0 | 0.256413 |
| prefix lambda_fast | early_log_R_q90_1 | 0.137737 |

完整排名见各目录中的 <code>feature_importance.csv</code>；它们说明不同 target/输入设定
依赖的物理摘要并不相同。

### Task D

当前 Task D 的 initial-condition 主实验预测半长轴 Lyapunov proxy：

```bash
conda run -n stellar python -m tasks.task_d \
  --task-setting initial \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --group-size 32 \
  --threshold 10000 \
  --target-mode lyapunov_proxy \
  --label-observable smax \
  --aggregation min \
  --censoring-rule all_pairs \
  --feature-aggregation first \
  --test-size 0.2 \
  --seed 42 \
  --output-dir results/task_d_initial \
  --completion-table-name completed_systems_sampling_2.txt \
  --prefix-len 200 \
  --prefix-time-stride 1 \
  --sequence-aggregation mean_std \
  --mlp-epochs 80 \
  --cnn-epochs 80 \
  --batch-size 128 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.1 \
  --val-fraction 0.1 \
  --patience 12 \
  --loss huber \
  --huber-beta 0.5
```

Prefix setting：

```bash
conda run -n stellar python -m tasks.task_d \
  --task-setting prefix \
  --target-mode lyapunov_proxy \
  --label-observable smax \
  --dataset-dir dataset_v3 \
  --sampling-id 2 \
  --group-size 32 \
  --threshold 1000 \
  --aggregation min \
  --censoring-rule all_pairs \
  --prefix-len 200 \
  --prefix-time-stride 1 \
  --sequence-aggregation mean_std \
  --output-dir results/task_d_prefix_tobs_200 \
  --mlp-epochs 80 \
  --cnn-epochs 80 \
  --batch-size 128 \
  --loss huber \
  --huber-beta 0.5
```

常用参数：

- `--force-recompute`: 忽略缓存标签并重新计算 proxy
- `--target-mode lyapunov_proxy`: 预测 `R_ij(t) >= threshold` 的首次采样时间；这是默认模式
- `--threshold`: 主实验默认 `10000`；建议用 sensitivity runner 同时报告 `1000/3000/10000/30000`
- `--aggregation min|median|mean`: 聚合 16 个 pair 的 proxy time；默认 `min`
- `--target-mode fitted_lyapunov_time`: 使用增长 slope 倒数的旧目标
- `--lyapunov-aggregation max|median|mean`: 控制 Lyapunov pair slope 的聚合方式
- `--require-pair-consistency`: 恢复 pair 间 MAD 一致性筛选
- `--lambda-r2-min`, `--min-valid-lambda-pairs`, `--max-pair-log-mad`: 调整 slope 与 pair consistency 质量筛选
- `--feature-aggregation first|mean_std`: initial setting 的 feature 聚合方式
- `--no-mlp`, `--no-cnn-orbital`, `--no-cnn-phase-space`, `--no-set-cnn`: 关闭对应神经模型
- `--loss mse|huber`: 神经网络回归损失
- `--allow-incomplete-valid-group`: 放宽完整 `32` 条有效 variation 的要求
