# Preference Reward

室内图生图用户偏好奖励模型项目。

## 当前阶段

当前已完成从数据处理到 Qwen3-VL 偏好奖励模型训练、独立推理和外部 GOOD/BAD 成对评估的完整链路。

模型输入为空房间原图和生成家具图，模型输出用户偏好奖励分数：

```text
reward_score = logit(A_like) - logit(B_dislike)
```

当前最佳模型为 Epoch 8，主要用于预测历史用户点赞/点踩偏好，并支持同一房间多个生成结果之间的排序。

## 目录说明

* `configs/`：数据、训练和模型配置文件
* `data/raw/`：原始 CSV，只读取不修改
* `data/images/`：本地图片数据，不提交到 Git
* `data/processed/`：清洗后的统一数据
* `data/splits/`：训练集和验证集
* `data/reports/`：字段、缺失值、标签和清洗统计
* `data/external/`：外部诊断测试清单；原始测试图片不提交
* `src/preference_reward/common/`：配置、日志和通用文件操作
* `src/preference_reward/data/`：数据结构、数据构建、清洗、划分和 Pairwise 清单读取
* `src/preference_reward/models/`：Qwen A/B Reward 模型封装
* `src/preference_reward/training/`：训练器和调度器
* `src/preference_reward/inference/`：Checkpoint 读取、模型加载和奖励打分
* `src/preference_reward/evaluation/`：分类评估和 GOOD/BAD 成对评估
* `scripts/`：命令行入口
* `tests/`：单元测试
* `logs/`：运行日志，不提交到 Git
* `outputs/`：模型输出、Checkpoint 和评估结果，不提交到 Git

## 核心设计

1. 原始数据只读。
2. 数据清洗、划分、训练、推理和评估模块职责分离。
3. 训练 Backend 不负责加载已训练 Checkpoint 或执行外部评估。
4. 推理统一通过独立 `inference/` 模块读取 Checkpoint、加载 LoRA Adapter 并计算 Reward。
5. Reward 定义为 `logit(A_like) - logit(B_dislike)`。
6. 保留 BF16 主模型计算，并对 A/B 输出方向执行 FP32 投影，避免奖励分数离散。
7. 外部 GOOD/BAD 成对数据仅用于独立诊断，不用于模型选择或反复调参。
8. 启动脚本只解析参数并调用模块，不重复实现底层逻辑。

## 当前主要能力

* 构建点赞/点踩偏好数据集
* 按房间隔离训练集和验证集
* 使用 Qwen3-VL-8B-Instruct + LoRA 训练 A/B 偏好奖励模型
* 保存可独立推理的 Checkpoint 配置
* 加载最佳 Epoch 的 LoRA Adapter 进行 Reward 打分
* 对外部 GOOD/BAD 测试集进行 Pairwise Accuracy、Bootstrap CI 和错误样本分析
* 生成 `REFERENCE | GOOD | BAD` 三联图用于人工审计

## 重要结果

当前最佳模型：

```text
Qwen3-VL-8B-Instruct + LoRA
Epoch 8
```

验证集结果：

```text
ROC-AUC ≈ 0.6254
```

外部 39 组成对诊断结果：

```text
Pairwise Accuracy ≈ 0.5897
```

该外部结果只能表述为弱正向排序趋势，不能作为显著优于随机的结论。

## OOF 泛化诊断与人工审计

为评估偏好模型在未参与训练样本上的泛化稳定性，我们在 749 条目标训练数据上进行了 4 折 Out-of-Fold（OOF）训练与预测。

每一折的 Outer Holdout 均不参与模型训练、学习率调度、早停和 Checkpoint 选择。

### OOF 结果

| Fold | Outer ROC-AUC |
|---|---:|
| 1 | 0.5553 |
| 2 | 0.5732 |
| 3 | 0.6181 |
| 4 | 0.5088 |

合并后的 OOF ROC-AUC：**0.5615**。

各 Fold 的结果波动较大，其中 Fold 4 接近随机排序，说明当前模型和数据尚未形成稳定的样本外偏好排序能力。

### 人工审计

从每个 Fold 的正负样本中分别抽取标签置信度较低的样本，共人工复核 80 条高冲突数据。

| 分类 | 含义 | 数量 | 比例 |
|---|---|---:|---:|
| A | 原始标签可能错误 | 18 | 22.5% |
| B | 合理的主观或少数偏好 | 17 | 21.2% |
| C | 模型判断错误，原始标签更合理 | 34 | 42.5% |
| D | 无法判断或样本异常 | 11 | 13.8% |

这些样本来自模型高冲突区域，因此上述比例不能直接解释为完整数据集的真实错标率。

### 当前结论

1. 数据中存在一定标签噪声，但标签错误不是唯一问题。
2. 在人工审计的高冲突样本中，模型误判是数量最多的类别。
3. 用户主观偏好差异也是不可忽略的误差来源。
4. 当前 OOF 预测不适合用于自动翻转标签、删除样本或设置训练权重。
5. 后续优先扩充严格住宅室内数据，同时保留目标数据用于最终对齐和独立评估。
