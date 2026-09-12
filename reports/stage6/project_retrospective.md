# 项目复盘说明

报告日期：2026-09-12

## 1. 做得比较好的部分

### 1.1 先固定数据和时间口径，再进入建模

项目没有从模型开始，而是先明确：

- 原始字段；
- 清洗规则；
- 历史窗口；
- 标签日；
- train / validation / test 时间顺序。

这一点直接降低了未来信息泄露风险。

Stage 3 所有规则都坚持：

```text
fit on train
apply to validation/test
```

并记录了窗口重叠为 0。

### 1.2 数据问题没有被简单“自动清洗”

Stage 1 发现大量重复行为，但时间只有小时粒度。

项目没有把“重复键”简单等同于异常，而是采用高频阈值策略并保留正常频次重复行为。这比无条件 `drop_duplicates()` 更符合当前数据可解释范围。

### 1.3 模型选择和最终测试严格分开

调参和 Stacking 只读取 train/validation。

测试集只在候选方案确定后做最终评估，避免了通过测试集反复试模型造成隐性过拟合。

### 1.4 极端类别不平衡被显式纳入设计

项目没有只训练一个模型，而是比较：

- baseline；
- SMOTE；
- undersampling；
- class weight。

这使得模型结果的核心 trade-off 非常清楚：

- XGBoost 排序和 LogLoss 更好；
- 欠采样模型 recall 很高，但误报非常大。

### 1.5 从模型指标走到了可解释性和推理

除了 AUC，项目还保留：

- PR / ROC；
- 混淆矩阵；
- 阈值敏感性；
- SHAP；
- FP/FN；
- 冷启动/长尾分群；
- 本地批量推理。

因此最终结果不是停留在 notebook 的单个分数。

---

## 2. 遇到的主要问题

### 2.1 数据规模大，不能依赖一次性全量 DataFrame

原始行为超过 1,225 万行，特征宽表超过 438 万行。

实际解决方式包括：

- 分块 CSV；
- Parquet；
- Hash 分区；
- 特征统计预计算；
- 看板读取轻量统计结果；
- 批量推理。

### 2.2 极端类别不平衡导致“准确率”非常容易误导

test 正样本率只有 0.1867%。

最终 XGBoost 在 0.5 阈值下：

```text
accuracy ≈ 99.81%
recall = 0
```

如果只看 accuracy，会得到完全错误的业务判断。

项目后期转而强调：

- AUC；
- Average Precision；
- Recall / Precision；
- 阈值敏感性；
- 概率排序。

### 2.3 默认分类阈值与业务目标不匹配

Stage 5 完整 test 推理最大概率约 0.232335，因此 0.5 阈值没有任何正类。

这说明模型“能排序”和“默认阈值能直接分类”是两件不同的事情。

未来应把阈值选择独立为业务步骤，以成本、收益和容量为依据。

### 2.4 模型序列化对环境版本敏感

本地加载 Stage 4 `joblib` 模型时，`scikit-learn` 版本不一致会产生 `InconsistentVersionWarning`。

解决方式是把推理环境 `scikit-learn` 固定到模型对应的 `1.8.0`。

后续更成熟的模型交付应考虑：

- 完整 environment lock；
- 模型 artifact metadata；
- XGBoost 原生模型格式；
- Docker / 容器化推理环境。

### 2.5 GPU 训练模型与 CPU 推理存在设备提示

Stage 5 推理时 XGBoost 提示模型/device 与输入不一致，最终自动回退 CPU DMatrix 并成功完成。

虽然不影响结果，但说明训练与推理设备配置应该显式化，以减少不必要的性能损失和告警。

### 2.6 模型文件不在 GitHub，需要单独的 artifact 交付约定

GitHub 正确地没有保存大模型，但这也意味着新环境无法仅靠 `git clone` 完成 Stage 5。

最终通过固定路径、bundle key、check-only 和文档解决。

未来建议增加：

```text
artifact manifest
checksum
training environment version
model creation commit SHA
```

让模型交付更可追踪。

---

## 3. 如果重新做一次，会改进什么

### 3.1 更早定义 Artifact Contract

在 Stage 3/4 开始时就固定：

```text
data contract
feature contract
model contract
inference contract
```

这样 Stage 5 不需要再从报告和本地文件中反查最终模型位置。

### 3.2 把概率校准和业务阈值纳入正式评估

当前项目已经有阈值敏感性，但没有单独做概率校准。

后续可以加入：

- calibration curve；
- Brier score；
- Platt scaling / isotonic calibration；
- Top-K precision / recall；
- lift / gain；
- 每千次触达预期收益。

这些指标比默认 0.5 分类更贴近营销业务。

### 3.3 单独处理长尾商品和冷启动

当前长尾商品正样本极少且 recall 为 0。

下一版可增加：

- 类目级 backoff；
- 商品 embedding；
- 内容侧特征；
- 时间衰减；
- exploration；
- 分群模型。

### 3.4 加入数据/特征漂移监控

训练、验证、测试的正样本率从：

```text
0.0304% → 0.0798% → 0.1867%
```

明显变化。

即使时间窗口不同造成部分差异，这也提示线上部署前需要监控：

- 标签基准率；
- 特征分布；
- 预测概率分布；
- AUC / AP；
- calibration。

### 3.5 更早做 CI

项目已经有较多功能测试和性能测试。

下一步可把不依赖大数据/模型的测试放到 GitHub Actions，每次 push 自动跑：

```text
unit / functional smoke
lint
small synthetic performance test
```

本地大数据和 GPU 测试继续保留离线执行。

---

## 4. 工程经验总结

### 数据工程

- 原始数据要只读；
- 数据质量检查与清洗要分开；
- 大数据处理优先使用 Parquet 和批处理；
- 缺失值要区分“业务合理缺失”和脏数据。

### 特征工程

- 时间窗口是最重要的边界；
- 训练集拟合规则必须与 validation/test 隔离；
- 高相关字段不只是“冗余”，还会让模型解释变得更困难；
- 行为序列、行为新鲜度比单纯计数更有信息。

### 建模

- 极端不平衡下不要以 accuracy 作为主指标；
- 高 recall 与高业务成本之间必须权衡；
- 融合模型不一定优于最佳单模型；
- 测试集必须真正留到最后。

### 部署

- 模型文件、特征清单、预处理规则必须版本一致；
- 推理代码不应隐式重新拟合任何训练阶段规则；
- 概率输出通常比固定阈值布尔输出更可复用；
- 模型二进制和大预测文件不适合直接存 GitHub。

---

## 5. 最终复盘

本项目最有价值的部分不是某一个模型，而是把：

```text
数据口径
→ 特征
→ 标签
→ 时间划分
→ 模型选择
→ 独立测试
→ 解释
→ 推理
```

串成了一条可以审计和复现的链路。

最终 XGBoost 在独立测试集 AUC 0.874953，说明已有特征能够有效排序购买倾向；同时默认阈值、长尾商品和极端类别不平衡又清楚展示了模型距离真实业务策略仍有的差距。

如果继续推进，下一阶段的重点不应是“再堆一个模型”，而应是：

1. 业务阈值与 Top-K 策略；
2. 概率校准；
3. 长尾/冷启动专用机制；
4. 数据漂移监控；
5. 模型 artifact 和环境版本管理；
6. 在线或准实时推理服务。
