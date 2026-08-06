# Task A short-horizon stable-fraction grid

本批实验使用与 Task A horizon grid 相同的 system-level split 和输入表示：

- dataset: `dataset_v3`, sampling `2`, group size `32`
- input: `orbital15` 初始条件，共 15 个特征
- split: `70% train / 10% validation / 20% test`, `seed=42`
- models: MLP-base and XGBoost-base
- target: `stable_fraction`
- device: CPU（当前环境没有可用的 NVIDIA driver）

| Horizon | Label mean | MLP MSE | MLP MAE | MLP R² | XGB MSE | XGB MAE | XGB R² |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.745861 | 0.051570 | 0.146435 | 0.599556 | **0.049802** | 0.155947 | **0.613285** |
| 200 | 0.682454 | 0.063825 | 0.179518 | 0.562286 | **0.059510** | **0.178483** | **0.591881** |
| 500 | 0.587529 | 0.076093 | **0.201114** | 0.536037 | **0.074599** | 0.211459 | **0.545143** |
| 1000 | 0.513718 | 0.088373 | **0.216604** | 0.482520 | **0.086658** | 0.234104 | **0.492558** |
| 2000 | 0.445616 | 0.090029 | **0.224702** | 0.469801 | **0.089360** | 0.237237 | **0.473743** |

每个 horizon 的完整产物位于对应的 `stable_fraction_hXXX/` 子目录，包括
`metrics.json`、模型文件、训练日志和 MLP 测试散点图。
