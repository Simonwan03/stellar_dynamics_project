# Experiment Results Summary (5/14)

This file reorganizes the current experimental results into a cleaner and more readable markdown summary.

## At A Glance

| Task | Best Model | Main Result | Short Takeaway |
| --- | --- | --- | --- |
| Task A | XGB / MLP are very close | XGB: `F1 0.8478`, MLP: `AUC 0.8489` | Both models are competitive; MLP is slightly better on ranking, XGB is slightly better on classification threshold metrics. |
| Task B | TCN | `AUC 0.9711`, `F1 0.8577`, `Acc 0.9067` | Early-prefix trajectory classification is strong across all temporal models. |
| Task C | Dual Attention | `overall_mae 0.0627`, `overall_rmse 0.1174`, `count_mae 2.0068` | Modeling the 8-trajectory structure clearly improves multi-horizon survival prediction. |
| Task D | CNN / RandomForest | CNN: best `MAE 0.833`, best `Spearman 0.772`; RF: best `RMSE 1.113` | Predicting Lyapunov-style time scales is feasible, but model ranking depends on metric choice. |

---

## Task A

Task A results are summarized below.

### Core Metrics

| Model | AUC | AP | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| XGB | `0.8458` | `0.9045` | `0.8478` | `0.7900` |
| MLP | `0.8489` | `0.9029` | `0.8434` | `0.7890` |

### Confusion Matrices

| Model | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| XGB | `205` | `134` | `76` | `585` |
| MLP | `221` | `118` | `93` | `568` |

### Takeaway

- XGB and MLP are very close overall.
- MLP has the best `AUC`, so its ranking quality is slightly stronger.
- XGB has slightly better `AP`, `F1`, and `Accuracy`, so its thresholded classification performance is slightly stronger on this run.

---

## Task B

Task B evaluates early-prefix trajectory classification.

### Core Metrics

| Model | Training Note | Test AUC | Test F1 | Test Accuracy |
| --- | --- | ---: | ---: | ---: |
| TCN | early stopping at epoch `27`, best val AUC `0.9711` | `0.9711` | `0.8577` | `0.9067` |
| CNN | no early-stopping note recorded here | `0.9696` | `0.8490` | `0.8986` |
| Attention | early stopping at epoch `39`, best val AUC `0.9691` | `0.9689` | `0.8548` | `0.9039` |

### CNN Debug Output

```text
pred_prob_test shape: (19994,)
pred_prob_test[:10]: [0.0944 0.     0.     0.5178 0.9505 0.1744 0.     0.4185 0.7675 0.    ]
pred_label_test[:10]: [0 0 0 1 1 0 0 0 1 0]
true_label_test[:10]: [0. 0. 0. 0. 1. 0. 0. 1. 0. 0.]
```

### Takeaway

- All three temporal models perform strongly, with test `AUC` near `0.97`.
- TCN is the best model on this run across `AUC`, `F1`, and `Accuracy`.
- Attention remains highly competitive and very close to TCN.

---

## Task C

Task C evaluates multi-horizon survival prediction on ensemble fraction
`Q = mean_i p_i`.

### Figures

![Task C Figure 1](image.png)

![Task C Figure 2](image-1.png)

### Overall Metrics

| Model | Loss | Trajectory BCE | Overall MAE | Overall RMSE | Count MAE | AUC MAE | Monotone Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean baseline | `0.8851` | `0.8851` | `0.3735` | `0.4090` | `11.9512` | `0.3616` | `1.0` |
| Trajectory context | `0.3023` | `0.3023` | `0.0748` | `0.1390` | `2.3921` | `0.0783` | `1.0` |
| Dual attention | `0.2704` | `0.2704` | `0.0627` | `0.1174` | `2.0068` | `0.0657` | `1.0` |
| Direct case-level | `0.0612` | `NaN` | `0.0768` | `0.1449` | `2.4567` | `0.0755` | `1.0` |

### Horizon-Specific Metrics

| Model | Final MAE | Early MAE | Mid MAE | Late MAE | Mean Bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean baseline | `0.3480` | `0.3926` | `0.3743` | `0.3535` | `-0.0038` |
| Trajectory context | `0.1076` | `0.0491` | `0.0733` | `0.1018` | `0.0446` |
| Dual attention | `0.0947` | `0.0382` | `0.0613` | `0.0886` | `0.0323` |
| Direct case-level | `0.0857` | `0.0703` | `0.0761` | `0.0839` | `-0.0023` |

### Takeaway

- The mean baseline is much worse than all learned models, so the early trajectory prefix contains strong predictive signal.
- `Dual attention` is the best overall model on the main aggregate metrics: `overall_mae`, `overall_rmse`, `count_mae`, and `auc_mae`.
- `Direct case-level` is not the best overall, but it achieves the best `final_mae`, so it is relatively strong at the final horizon specifically.
- All models satisfy the monotonicity constraint perfectly: `monotone_fraction = 1.0`.
- The biggest jump comes from explicitly modeling structured trajectory information rather than relying on a trivial baseline.

---

## Task D

Task D predicts `log_tau_system`, where:

- `delta_a(t) = abs(a_i(t) - a_j(t))`
- `R_a(t) = delta_a(t) / delta_a(0)`
- instability threshold: `R_a(t) > 1e3`
- `tau_system = median(pair_taus)`
- `log_tau_system = np.log1p(tau_system)`

### Core Metrics

| Model | MAE(log_tau) | RMSE(log_tau) | Spearman |
| --- | ---: | ---: | ---: |
| RandomForest | `0.878` | `1.113` | `0.722` |
| MLP | `1.013` | `1.295` | `0.607` |
| CNN | `0.833` | `1.561` | `0.772` |

### Notes

- For RandomForest, `MAE(log_tau) = 0.878`, corresponding to an approximate multiplicative factor `exp(0.878) ≈ 2.4`.

### Takeaway

- CNN achieves the best `MAE(log_tau)` and the best `Spearman`, so it is strongest on absolute error and rank correlation.
- RandomForest achieves the best `RMSE(log_tau)`, so it is more robust on large-error penalties under this metric.
- MLP is currently the weakest of the three models on all reported metrics.

---

## Overall Conclusions

1. Task A is already competitive with relatively standard models, and XGB/MLP are nearly tied.
2. Task B shows that early trajectory prefixes are highly predictive of future outcomes; TCN is currently the strongest model in this setting.
3. Task C is the clearest success case: structured multi-trajectory modeling substantially improves survival prediction, and the monotonic survival constraint is always satisfied.
4. Task D is promising but less settled: CNN leads on `MAE` and rank correlation, while RandomForest leads on `RMSE`, so the preferred model depends on the downstream objective.
