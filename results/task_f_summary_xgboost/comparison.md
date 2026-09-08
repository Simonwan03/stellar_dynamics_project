# Task F: Task-B dynamical summaries + XGBoost comparison

Metrics use the same Task-F test split and deterministic test mask for each ratio.
The summary features use only observed positions; initial/final are the first/last observed values.

| mask_ratio | model | accuracy@0.5 | balanced_accuracy@0.5 | f1@0.5 | f1@val_threshold | auc | average_precision | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0% | Attention | 0.7701 | 0.6775 | 0.5391 | 0.6752 | 0.8534 | 0.6976 | 0.1680 |
| 0% | Summary + XGBoost | 0.8889 | 0.8610 | 0.8143 | 0.8239 | 0.9549 | 0.9037 | 0.0788 |
| 0% | TCN | 0.7927 | 0.7447 | 0.6487 | 0.6628 | 0.8559 | 0.7146 | 0.1526 |
| 25% | Attention | 0.7930 | 0.7418 | 0.6448 | 0.7016 | 0.8725 | 0.7191 | 0.1442 |
| 25% | Summary + XGBoost | 0.8885 | 0.8601 | 0.8133 | 0.8194 | 0.9543 | 0.9002 | 0.0794 |
| 25% | TCN | 0.7874 | 0.7466 | 0.6504 | 0.6692 | 0.8549 | 0.6944 | 0.1493 |
| 50% | Attention | 0.7930 | 0.7437 | 0.6474 | 0.6860 | 0.8658 | 0.7132 | 0.1437 |
| 50% | Summary + XGBoost | 0.8872 | 0.8594 | 0.8118 | 0.8220 | 0.9539 | 0.8993 | 0.0799 |
| 50% | TCN | 0.7777 | 0.7555 | 0.6599 | 0.6671 | 0.8494 | 0.6867 | 0.1543 |
| 75% | Attention | 0.7779 | 0.7540 | 0.6581 | 0.6683 | 0.8563 | 0.6973 | 0.1479 |
| 75% | Summary + XGBoost | 0.8846 | 0.8550 | 0.8065 | 0.8218 | 0.9527 | 0.8994 | 0.0810 |
| 75% | TCN | 0.7764 | 0.7095 | 0.5963 | 0.6536 | 0.8395 | 0.6816 | 0.1638 |
