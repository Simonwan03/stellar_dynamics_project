当前仓库有两套 split 口径：
1. Task A/B/C：共用同一份 system-level split
2. Task E：也按 system 分组，但使用自己的可预测子集和独立随机 split
Task A/B/C 公共 Split
manifest：
dataset_v3/common_case_split_sampling_2_g32_h10000_r70-15-15_seed42.csv
配置：
- 数据集：dataset_v3/sampling_2
- 一个 system 包含 32 个 variations
- raw_case_id = run_id // 32
- 以 system 为切分单位，避免同一 system 的 variations 泄漏到不同集合
- seed=42
- 比例：70% / 15% / 15%
- 时间点：time_horizon=10000
- 有效全集：4998 个完整 system，共 159936 条 trajectory
Task / 模式	Train	Val	Test	切分单位
Task A	3498 systems	750 systems	750 systems	system
Task B trajectory_binary	111936 trajectories	24000 trajectories	24000 trajectories	system group 内的 trajectory
Task B group_fraction	3498 systems	750 systems	750 systems	system
Task C	3498 systems	750 systems	750 systems	system
Task A
Task A 虽然输入文件保留了每个 system 的 32 个初始条件，但默认建模时使用每组的第一个 variation，因此实际训练样本是：
train: 3498
val:    750
test:   750
stable_fraction 和 binary_any 两种目标使用完全相同的 split。
Task B
- trajectory_binary：每条 trajectory 是一条样本，因此 system 数乘以 32。
- group_fraction：32 条 trajectory 聚合为一个 system 样本，因此恢复为 3498/750/750。
- prefix 长度变化，例如 96、100、512，不会改变公共 system split。
Task C
每个样本是一个完整 system，输入通常为：
[32 variations, prefix_length, channels]
variation ablation 只改变每个 system 输入多少条 variations，不改变 train/val/test 的 system 边界。
公共 split 还按 10000 时刻的 stable fraction 做粗分层：
0
(0, 0.25]
(0.25, 0.5]
(0.5, 1)
1
这只用于平衡各集合的稳定性分布，不改变任务标签。
Task E Split
当前主协议使用：
orbital15
context = 512
prediction = 128
stride = 20
Task E 不直接读取 A/B/C 的公共 manifest。它先进行 completion-aware filtering，再在可构造完整 512+128=640 步窗口的 trajectory 所属 system 上独立随机切分。
过滤后的实际规模
原始数据：
5000 systems
160000 trajectories
completion table 过滤后：
4998 complete systems
159936 valid trajectories
但由于要求每条 trajectory 至少能形成完整的 640-step forecasting window，最终进入 Task E split 的是：
3847 systems
64597 trajectories
当前 E1/E2 运行中的实际 split 为：
Split	Systems	Trajectories
Train	2692	44988
Val	577	9961
Test	578	9648
Total	3847	64597
因此，Task E 中同一 system 仍不会跨 split，但一个 system 在过滤后不一定保留全部 32 条 trajectory。
Task E 窗口数量
当前 512 -> 128 主运行的 candidate windows：
Split	Candidate windows
Train	2,678,116
Val	600,998
E1 Test	579,577
实际使用时不是全部窗口：
- Train：每条 trajectory 每个 epoch 随机采样最多 8 个窗口，约 351944 个/epoch
- Val：每条 trajectory 固定采样最多 4 个窗口，共 39518
- E1 Test：固定采样最多 4 个窗口，共 38245
- E2 rolling：每条 trajectory 采样最多 2 个 rolling start，共 17485
Scaler 只使用 train split 中、失稳事件之前且数值有限的窗口拟合。
重要区别
虽然 Task E 也使用 seed=42 和 70/15/15，但它的 system 集合已经经过 forecasting-window filtering，而且没有使用 A/B/C 的稳定比例分层。因此：
Task A/B/C 的 system membership
不等于
Task E 的 system membership
另外，部分旧结果目录仍使用旧的 3498/500/1000 split，例如某些 Task C 结果；这些不能和当前 3498/750/750 公共 split 的结果直接混合比较。