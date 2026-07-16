# Preference Reward

室内图生图用户偏好奖励模型项目。

## 当前阶段

当前已完成从数据处理到 Qwen3-VL 偏好奖励模型训练、独立推理和外部 GOOD/BAD 成对评估的完整链路。

模型输入为空房间原图和生成家具图，模型输出用户偏好奖励分数：

```text
reward_score = logit(A_like) - logit(B_dislike)
```

当前最佳参考模型为无房型 Qwen3-VL-8B，最佳 Epoch 为 8，主要用于预测历史用户点赞/点踩偏好，并支持同一房间多个生成结果之间的排序。

## 目录说明

- `configs/`：数据、训练和模型配置文件
- `data/raw/`：原始 CSV，只读取不修改
- `data/images/`：本地图片数据，不提交到 Git
- `data/processed/`：清洗后的统一数据
- `data/splits/`：训练集和验证集
- `data/reports/`：字段、缺失值、标签和清洗统计
- `data/external/`：外部诊断测试清单；原始测试图片不提交
- `src/preference_reward/common/`：配置、日志和通用文件操作
- `src/preference_reward/data/`：数据结构、数据构建、清洗、划分和 Pairwise 清单读取
- `src/preference_reward/models/`：Qwen A/B Reward 模型封装和 Prompt 处理
- `src/preference_reward/training/`：训练器、调度器和独立正则损失
- `src/preference_reward/inference/`：Checkpoint 读取、模型加载和奖励打分
- `src/preference_reward/evaluation/`：分类评估、GOOD/BAD 成对评估和房型诊断
- `scripts/`：训练、数据构建和独立评估入口
- `tests/`：单元测试
- `logs/`：运行日志，不提交到 Git
- `outputs/`：模型输出、Checkpoint 和评估结果，不提交到 Git

### 本阶段新增结构

```text
configs/
├── qwen8b_layout_512_room_type.yaml
└── qwen8b_layout_512_margin_l2.yaml

src/preference_reward/
├── data/
│   └── room_type_manifest.py
├── models/
│   └── prompting.py
├── training/
│   └── margin_regularization.py
└── evaluation/
    └── room_type.py

scripts/
├── add_room_type_to_manifests.py
├── evaluate_room_type.py
├── evaluate_training_manifest.py
└── build_layout_test100_manifests.py

tests/
├── test_room_type_manifest.py
├── test_room_type_prompt.py
├── test_room_type_checkpoint.py
├── test_room_type_evaluation.py
├── test_evaluate_room_type_script.py
└── test_margin_regularization.py
```

## 核心设计

1. 原始数据只读。
2. 数据清洗、划分、训练、推理和评估模块职责分离。
3. 训练 Backend 不负责加载已训练 Checkpoint 或执行外部评估。
4. 推理统一通过独立 `inference/` 模块读取 Checkpoint、加载 LoRA Adapter 并计算 Reward。
5. Reward 定义为 `logit(A_like) - logit(B_dislike)`。
6. 保留 BF16 主模型计算，并对 A/B 输出方向执行 FP32 投影，避免奖励分数离散。
7. 外部 GOOD/BAD 成对数据仅用于独立诊断，不用于模型选择或反复调参。
8. 启动脚本只解析参数并调用模块，不重复实现底层逻辑。
9. 新增功能优先放在独立模块中，训练器只保留最小调用接缝。

## 当前主要能力

- 构建点赞/点踩偏好数据集
- 按房间隔离训练集和验证集
- 使用 Qwen3-VL-8B-Instruct + LoRA 训练 A/B 偏好奖励模型
- 保存可独立推理的 Checkpoint 配置
- 加载最佳 Epoch 的 LoRA Adapter 进行 Reward 打分
- 对外部 GOOD/BAD 测试集进行 Pairwise Accuracy、Bootstrap CI 和错误样本分析
- 生成 `REFERENCE | GOOD | BAD` 三联图用于人工审计
- 执行四折 OOF 泛化评估
- 执行房型捷径诊断
- 评估最佳 Checkpoint 在完整训练集上的表现
- 比较两个模型在同一批 Pair 上的差异

## 重要结果

当前最佳参考模型：

```text
Qwen3-VL-8B-Instruct + LoRA
无房型
Epoch 8
```

固定验证集结果：

```text
ROC-AUC ≈ 0.6254
```

外部 39 组成对诊断结果：

```text
Pairwise Accuracy ≈ 58.97%
```

扩展 100 组成对诊断结果：

```text
Pairwise Accuracy = 56.00%
```

该外部结果只能表述为弱正向排序趋势，不能作为显著优于随机的结论。

## OOF 泛化诊断

固定验证集上的最佳 ROC-AUC 约为 `0.6254`，但该结果来自单次固定划分，不能完全反映模型在未参与训练样本上的稳定泛化能力。

在 749 条目标训练数据上执行了 4 折 Out-of-Fold 训练与预测。每一折的 Outer Holdout 均不参与模型训练、学习率调度、早停和 Checkpoint 选择。

| Fold | Outer ROC-AUC |
|---|---:|
| 1 | 0.5553 |
| 2 | 0.5732 |
| 3 | 0.6181 |
| 4 | 0.5088 |

合并后的 OOF ROC-AUC：

```text
0.5615
```

四折结果差异较大，最差一折接近随机，说明模型效果依赖数据划分，稳定性不足。

## 房型信息测试

在无房型 Baseline 基础上加入房型 Prompt，最佳验证集 ROC-AUC 从 `0.6254` 小幅提高到 `0.6291`。

但外部 39 组结果明显下降：

| 模型 | 正确数 | Pair Accuracy |
|---|---:|---:|
| 无房型 Baseline | 23 / 39 | 58.97% |
| 加入房型 | 17 / 39 | 43.59% |

进一步诊断结果：

```text
模型总体 AUC：       0.6291
房型内 Weighted AUC：0.5535
RoomType Only AUC：  0.6673
```

结论：

> 房型与点赞率之间存在明显相关性。模型容易通过房型猜测标签，而不是学习同一房型内部的布局质量，因此房型属于数据捷径，不再作为模型输入。

## Margin-L2 测试

在无房型 512 配置上增加限制 A/B 分数差过大的 Margin-L2 正则。

固定验证集：

| 模型 | ROC-AUC |
|---|---:|
| Baseline | 0.6254 |
| Margin-L2 | 0.6099 |

原 39 组测试：

| 模型 | 正确数 | Pair Accuracy |
|---|---:|---:|
| Baseline | 23 / 39 | 58.97% |
| Margin-L2 | 25 / 39 | 64.10% |

扩展 100 组测试：

| 模型 | 正确数 | Pair Accuracy |
|---|---:|---:|
| Baseline | 56 / 100 | 56.00% |
| Margin-L2 | 56 / 100 | 56.00% |

真正新增的 61 组：

| 模型 | 正确数 | Pair Accuracy |
|---|---:|---:|
| Baseline | 33 / 61 | 54.10% |
| Margin-L2 | 31 / 61 | 50.82% |

结论：

> Margin-L2 在原 39 组上的小幅提升没有在新增数据中复现，整体没有稳定收益，因此停止继续调整该方向。

## 训练集与泛化差距

无房型 Baseline 最佳 Epoch 8 在完整训练集上的结果：

| 指标 | 训练集 |
|---|---:|
| ROC-AUC | 0.9142 |
| Accuracy | 85.98% |
| Balanced Accuracy | 0.8119 |
| Positive PR-AUC | 0.9485 |
| Negative PR-AUC | 0.8684 |
| Brier Score | 0.1063 |
| ECE | 0.0474 |

训练集与验证集差距：

```text
Train ROC-AUC：0.9142
Val ROC-AUC：  0.6254
Gap：          0.2888
```

Margin-L2 的训练集 ROC-AUC 达到 `0.9579`，但验证集下降到 `0.6099`，Train–Validation Gap 扩大到 `0.3480`。

结论：

> 当前模型已经能够充分拟合训练数据，主要问题是泛化不足，而不是 LoRA Rank 或模型容量不足。

## AID Feedback V2 数据扩展结果

本阶段将可用用户反馈数据从上一版的 936 条扩大到 5,236 条。其中训练集 4,189 条，验证集 1,047 条：

```text
Train：Like 3,035 / Dislike 1,154
Validation：Like 759 / Dislike 288
```

当前最佳模型为：

```text
Qwen3-VL-8B-Instruct + LoRA
无房型
512 分辨率
Epoch 4
```

固定验证集结果：

| 指标                |     结果 |
| ----------------- | -----: |
| ROC-AUC           | 0.7047 |
| Balanced Accuracy | 0.5898 |
| Negative PR-AUC   | 0.5001 |
| Brier Score       | 0.1770 |
| ECE               | 0.0201 |

与上一版相比，固定验证集 ROC-AUC 从 `0.6254` 提高到 `0.7047`，绝对提升约 `0.0793`。

在扩展 100 组 `REFERENCE / GOOD / BAD` 数据上的结果：

```text
正确：66
错误：34
Pairwise Accuracy：66.00%
95% CI：[56.57%, 75.25%]
平均 GOOD-BAD Margin：0.1931
Pointwise ROC-AUC：0.5873
```

上一版模型在相同 100 组上的 Pairwise Accuracy 为 `56%`，本阶段提高到 `66%`。说明扩大真实用户反馈数据不仅改善了内部验证集表现，也提升了模型对人工 GOOD/BAD 布局结果的相对排序能力。

模型在 Epoch 4 达到最佳结果，之后训练损失继续下降，但验证指标逐步恶化，并在 Epoch 9 触发 Early Stopping，说明继续增加训练轮数不能解决过拟合问题。

当前模型更适合同一参考图下多个生成结果之间的候选排序，而不是使用统一阈值判断不同房间的绝对质量。下一阶段将使用现有 100 组 `R / GOOD / BAD` 数据，从 Qwen3-VL-8B 基模重新训练专项布局 Reward Model，并按 Pair 隔离训练集和验证集。
