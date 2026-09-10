# 阶段四 Issue4：模型可解释性与错误样本分析

## 分析对象与口径

- 分析模型：`baseline__xgboost`。该模型由 Issue2.5 的验证集 AUC 领先记录确定，并在 Issue3 中保留测试集结果。
- 错误样本阈值：`0.05`，取自 Issue3 中该模型 F1 最高的既有阈值网格结果。
- 本步骤只加载既有模型、测试集预测和特征数据；不训练新模型、不调参、不新增特征。

## 特征重要性 Top10

| feature | mean_abs_shap | mean_shap |
| --- | --- | --- |
| last_behavior_days_ago | 1.135047 | 1.135047 |
| last_10_behavior_sequence_code | 0.523118 | -0.292547 |
| item_cart_count | 0.403292 | -0.238763 |
| user_buy_count | 0.384174 | -0.371447 |
| user_unique_category_count | 0.292961 | 0.285887 |
| user_unique_item_count | 0.259256 | 0.254223 |
| category_fav_count | 0.214881 | 0.147517 |
| item_total_count | 0.174674 | -0.098584 |
| buy_per_cart | 0.132185 | -0.090116 |
| item_unique_user_count | 0.104432 | 0.081860 |

## 正负向预测依据

方向来自实际购买样本上的平均 SHAP 值：正值表示该特征在该样本群中倾向于抬高购买预测，负值表示倾向于降低预测。它反映模型关联，不代表因果关系。

### 正向影响 Top5

| feature | mean_shap_on_actual_positive |
| --- | --- |
| last_behavior_days_ago | 1.761834 |
| last_10_behavior_sequence_code | 0.429610 |
| user_unique_item_count | 0.314259 |
| user_unique_category_count | 0.287680 |
| item_cart_count | 0.224910 |

### 负向影响 Top5

| feature | mean_shap_on_actual_positive |
| --- | --- |
| user_buy_count | -0.345201 |
| category_buy_count | -0.094859 |
| time_weekday | -0.083244 |
| user_active_day_count | -0.033621 |
| item_buy_count | -0.025857 |

## 错误样本统计

| tn | fp | fn | tp |
| --- | --- | --- | --- |
| 328852 | 470 | 578 | 38 |

## 冷启动与长尾商品场景

冷启动用户定义为原始历史 `user_event_count <= 1`；长尾商品定义为原始历史 `item_total_count <= 1`。计数先按阶段三训练集拟合的标准化规则还原，两个定义只使用标签日前可得的历史统计特征。

| segment | sample_count | positive_count | positive_rate | precision | recall | f1 | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_test_samples | 329938 | 616 | 0.001867 | 0.074803 | 0.061688 | 0.067616 | 470 | 578 |
| cold_start_user | 102 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 |
| non_cold_start_user | 329836 | 616 | 0.001868 | 0.074803 | 0.061688 | 0.067616 | 470 | 578 |
| long_tail_item | 77428 | 26 | 0.000336 | 0.000000 | 0.000000 | 0.000000 | 1 | 26 |
| non_long_tail_item | 252510 | 590 | 0.002337 | 0.074951 | 0.064407 | 0.069280 | 469 | 552 |
| cold_start_and_long_tail | 76 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 |

详细 Top10、SHAP 全量重要性、假阳性/假阴性样例和分群统计见同目录 CSV 文件。
