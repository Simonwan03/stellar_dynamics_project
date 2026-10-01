| Prefix | Representation | Model | Trajectory Accuracy | Trajectory F1 | Trajectory AUROC | Trajectory AUPRC | System MAE ↓ | System MSE ↓ | System R² ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 96 | Panda encoder | Linear | 0.8640 | 0.8070 | 0.9467 | 0.8911 | 0.1031 | 0.0256 | 0.8390 |
| 96 | Chronos-2 encoder | Linear | 0.8623 | 0.8067 | 0.9531 | 0.9083 | 0.1112 | 0.0281 | 0.8230 |
| 96 | Dynamical summaries | XGBoost | 0.9000 | 0.8404 | 0.9649 | 0.9319 | 0.0703 | 0.0153 | 0.9040 |
| 96 | Raw sequence | SVM | 0.7493 | 0.5800 | 0.8135 | 0.6166 | 0.2373 | 0.0927 | 0.4165 |
| 96 | Raw sequence | CNN | 0.7937 | 0.6092 | 0.8888 | 0.7526 | 0.1759 | 0.0786 | 0.5056 |
| 96 | Raw sequence | TCN | 0.8167 | 0.7402 | 0.8971 | 0.7523 | 0.1570 | 0.0581 | 0.6347 |
| 512 | Panda encoder | Linear | 0.9217 | 0.8845 | 0.9786 | 0.9550 | 0.0534 | 0.0071 | 0.9554 |
| 512 | Chronos-2 encoder | Linear | 0.9209 | 0.8848 | 0.9791 | 0.9552 | 0.0614 | 0.0087 | 0.9452 |
| 512 | Dynamical summaries | XGBoost | 0.9329 | 0.8960 | 0.9851 | 0.9673 | 0.0398 | 0.0051 | 0.9676 |
| 512 | Raw sequence | SVM | 0.7590 | 0.5918 | 0.8282 | 0.6232 | 0.2184 | 0.0818 | 0.4855 |
| 512 | Raw sequence | CNN | 0.8839 | 0.8205 | 0.9470 | 0.8543 | 0.0985 | 0.0278 | 0.8249 |
| 512 | Raw sequence | TCN | 0.8973 | 0.8411 | 0.9593 | 0.8900 | 0.0765 | 0.0177 | 0.8886 |
