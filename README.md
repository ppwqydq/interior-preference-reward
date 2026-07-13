# Preference Reward

室内设计生成结果用户偏好模型项目。

## 当前阶段

当前只处理数据检查、数据清洗和训练集划分。

## 目录说明

- `configs/`：配置文件
- `data/raw/`：原始 CSV，只读取不修改
- `data/processed/`：清洗后的统一数据
- `data/splits/`：训练集和验证集
- `data/reports/`：字段、缺失值、标签和清洗统计
- `src/preference_reward/common/`：日志和通用文件操作
- `src/preference_reward/data/`：数据结构、检查、清洗和划分
- `scripts/`：命令行入口
- `tests/`：单元测试
- `logs/`：运行日志
- `outputs/`：后续模型输出

## 设计原则

1. 原始数据只读。
2. 清洗规则集中管理。
3. 所有被过滤记录都统计具体原因。
4. 训练和评估不得重复实现数据预处理。
5. 启动脚本只解析参数并调用模块。
