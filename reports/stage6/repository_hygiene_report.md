# GitHub 仓库与大文件检查报告

检查日期：2026-09-12

## 1. 检查目标

确认最终 GitHub 仓库：

- 目录结构清晰；
- `.gitignore` 能阻止数据、模型和预测大文件进入 Git；
- 当前跟踪内容没有大数据/模型二进制；
- 代码、配置、文档和轻量结果足以解释和复现项目流程。

## 2. 目录结构

核心目录职责清晰：

| 目录 | 用途 |
| --- | --- |
| `configs/` | 可提交的轻量运行配置 |
| `dashboards/` | Streamlit EDA / 模型看板 |
| `data/` | 本地/共享大数据，仅 Git 保留骨架 |
| `docs/` | 项目、接口、运行与阶段文档 |
| `models/` | 本地模型产物，仅保留 README / 骨架 |
| `outputs/` | 本地预测、性能和测试运行产物 |
| `reports/` | 可提交的轻量指标、图表和报告 |
| `scripts/` | 命令行与 pipeline 入口 |
| `src/` | 核心实现 |
| `tests/` | 功能与性能测试 |

## 3. `.gitignore` 检查

当前规则覆盖：

```text
data/**
outputs/**
artifacts/**
models/**
experiments/runs/**
experiments/tracking/**
```

同时忽略：

```text
*.csv
*.parquet
*.arrow
*.feather
*.db
*.sqlite*
```

Stage 4 需要进入报告的轻量 CSV 通过明确的 `!reports/stage4/...` 例外保留。

本次最终整理额外增加通用模型二进制保护：

```text
*.joblib
*.pkl
*.pickle
*.onnx
```

用于防止模型文件被误放到 `models/` 以外后进入 Git。

## 4. 当前仓库大文件检查

在最终整理所依据的跟踪文件快照中，没有发现大型数据或模型二进制。

最大的轻量跟踪文件约为：

| 文件 | 大小 |
| --- | ---: |
| `reports/stage3/preprocessing_rules.json` | 326,124 bytes |
| `reports/stage4/model_evaluation/test_roc_curves.png` | 101,021 bytes |
| `reports/stage4/model_evaluation/test_pr_curves.png` | 63,802 bytes |
| `reports/stage4/model_interpretability/shap_feature_importance_top10.png` | 54,804 bytes |

没有跟踪：

```text
*.parquet
*.joblib
```

等大数据/模型文件。

Stage 5 实际模型：

```text
models/traditional_ml/stacking/models/base_models.joblib
```

保留在本地/团队 artifact 存储，不进入 GitHub。

Stage 5 全量预测：

```text
outputs/predictions/stage5_test_predictions.parquet
```

同样保留本地，不进入 GitHub。

## 5. 推荐提交前检查

PowerShell：

```powershell
git status --short

git ls-files |
  ForEach-Object {
    if (Test-Path $_) { Get-Item $_ }
  } |
  Sort-Object Length -Descending |
  Select-Object -First 20 FullName,Length

git ls-files |
  Select-String -Pattern '\.(parquet|joblib|pkl|pickle|onnx|db|sqlite|zip)$'
```

最后一条理想情况下不应返回数据、模型或本地压缩包。

也可检查暂存区：

```powershell
git diff --cached --stat
git diff --cached --check
```

## 6. 结论

当前仓库符合“GitHub 保存代码和轻量结果、Google Drive/本地保存大数据和模型”的项目约定。

本次整理没有要求上传：

- 大型数据；
- 模型二进制；
- 全量预测结果。

仓库结构和 `.gitignore` 可以支持最终交付。
