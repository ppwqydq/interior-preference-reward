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
