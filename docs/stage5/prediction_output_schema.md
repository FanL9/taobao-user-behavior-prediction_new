# 阶段五预测结果字段说明

默认预测文件：

`outputs/predictions/purchase_probability_predictions.parquet`

| 字段 | 类型/范围 | 说明 |
| --- | --- | --- |
| `user_id` | 原输入类型 | 用户标识，仅用于追踪 |
| `item_id` | 原输入类型 | 商品标识，仅用于追踪 |
| `category_id` | 原输入类型 | 类目标识，仅用于追踪 |
| `purchase_probability` | float, `[0, 1]` | 最终模型的购买正类概率 |
| `predicted_purchase` | int8, `{0,1}` | 按阶段四固定阈值 0.5 二值化后的辅助字段 |

核心业务输出为 `purchase_probability`。`predicted_purchase` 仅用于固定阈值下的便捷查看，不替代概率排序。
