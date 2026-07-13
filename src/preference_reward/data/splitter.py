#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""训练集和验证集的分组划分。

划分原则：
1. 以空房间图片为分组单位。
2. 同一空房间的所有生成家具图必须位于同一集合。
3. 尽量保持验证集样本比例和正负标签比例接近全量数据。
4. 不从当前数据中划分测试集，测试集由外部布局数据提供。
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取并校验训练样本 JSONL。"""

    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()

            if not text:
                continue

            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON 解析失败：{path}:{line_number}"
                ) from exc

            required_fields = {
                "empty_room_image",
                "generated_furniture_image",
                "label",
            }

            missing_fields = required_fields - set(row)

            if missing_fields:
                raise KeyError(
                    f"{path}:{line_number} 缺少字段："
                    f"{sorted(missing_fields)}"
                )

            label = int(row["label"])

            if label not in (0, 1):
                raise ValueError(
                    f"{path}:{line_number} 标签无效：{label}"
                )

            row["label"] = label
            rows.append(row)

    if not rows:
        raise RuntimeError(f"数据文件为空：{path}")

    return rows


def group_rows(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """按照空房间图片对样本分组。"""

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        group_key = str(row["empty_room_image"])
        groups[group_key].append(row)

    return dict(groups)


def label_counts(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    """统计正负样本数量。"""

    counts = Counter(int(row["label"]) for row in rows)

    return {
        "0": int(counts.get(0, 0)),
        "1": int(counts.get(1, 0)),
    }


def positive_rate(
    rows: Sequence[Dict[str, Any]],
) -> float:
    """计算正样本比例。"""

    if not rows:
        return 0.0

    return sum(int(row["label"]) for row in rows) / len(rows)


def candidate_score(
    validation_rows: Sequence[Dict[str, Any]],
    total_rows: int,
    target_validation_size: int,
    overall_positive_rate: float,
) -> float:
    """评价一次候选划分。

    分数越低越好，同时考虑：
    - 验证集样本数量是否接近目标；
    - 验证集正样本比例是否接近全量数据。
    """

    size_error = (
        abs(len(validation_rows) - target_validation_size)
        / total_rows
    )

    ratio_error = abs(
        positive_rate(validation_rows)
        - overall_positive_rate
    )

    return size_error + ratio_error


def search_validation_groups(
    groups: Dict[str, List[Dict[str, Any]]],
    total_rows: int,
    validation_ratio: float,
    seed: int,
    search_trials: int,
) -> set[str]:
    """通过多次确定性随机搜索选择验证集分组。

    由于每个空房间对应的样本数量可能不同，无法保证精确切出
    固定行数。通过多次搜索选择最接近目标比例和标签比例的结果。
    """

    group_keys = sorted(groups)

    if len(group_keys) < 2:
        raise RuntimeError(
            "空房间分组少于 2 个，无法划分训练集和验证集"
        )

    target_validation_size = round(
        total_rows * validation_ratio
    )

    all_rows = [
        row
        for group_rows_list in groups.values()
        for row in group_rows_list
    ]

    overall_positive_rate = positive_rate(all_rows)

    best_group_keys: set[str] | None = None
    best_score: float | None = None

    for trial in range(search_trials):
        shuffled_keys = list(group_keys)

        # seed + trial 保证相同配置始终得到相同结果。
        random.Random(seed + trial).shuffle(shuffled_keys)

        selected_keys: set[str] = set()
        selected_rows: List[Dict[str, Any]] = []

        for group_key in shuffled_keys:
            if len(selected_rows) >= target_validation_size:
                break

            selected_keys.add(group_key)
            selected_rows.extend(groups[group_key])

        selected_labels = {
            int(row["label"])
            for row in selected_rows
        }

        # 验证集中必须同时存在正负样本。
        if selected_labels != {0, 1}:
            continue

        train_labels = {
            int(row["label"])
            for key, group_items in groups.items()
            if key not in selected_keys
            for row in group_items
        }

        # 训练集中也必须同时存在正负样本。
        if train_labels != {0, 1}:
            continue

        score = candidate_score(
            validation_rows=selected_rows,
            total_rows=total_rows,
            target_validation_size=target_validation_size,
            overall_positive_rate=overall_positive_rate,
        )

        if best_score is None or score < best_score:
            best_score = score
            best_group_keys = selected_keys

    if best_group_keys is None:
        raise RuntimeError(
            "未找到同时包含正负样本的有效分组划分"
        )

    return best_group_keys


def split_dataset(
    input_manifest: Path,
    train_output: Path,
    validation_output: Path,
    report_output: Path,
    validation_ratio: float = 0.2,
    seed: int = 42,
    search_trials: int = 5000,
) -> Dict[str, Any]:
    """执行分组划分并输出结果。"""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio 必须位于 0 和 1 之间"
        )

    if search_trials <= 0:
        raise ValueError(
            "search_trials 必须大于 0"
        )

    rows = read_jsonl(input_manifest)
    groups = group_rows(rows)

    validation_group_keys = search_validation_groups(
        groups=groups,
        total_rows=len(rows),
        validation_ratio=validation_ratio,
        seed=seed,
        search_trials=search_trials,
    )

    train_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []

    for group_key, group_items in groups.items():
        if group_key in validation_group_keys:
            validation_rows.extend(group_items)
        else:
            train_rows.extend(group_items)

    # 固定排序，保证输出文件可复现。
    sort_key = lambda row: (
        str(row["empty_room_image"]),
        str(row["generated_furniture_image"]),
        int(row["label"]),
    )

    train_rows.sort(key=sort_key)
    validation_rows.sort(key=sort_key)

    train_groups = {
        str(row["empty_room_image"])
        for row in train_rows
    }

    validation_groups = {
        str(row["empty_room_image"])
        for row in validation_rows
    }

    overlap_groups = train_groups & validation_groups

    if overlap_groups:
        raise RuntimeError(
            "检测到训练集和验证集空房间分组重叠"
        )

    write_jsonl_atomic(
        train_rows,
        train_output,
    )

    write_jsonl_atomic(
        validation_rows,
        validation_output,
    )

    report = {
        "input_manifest": str(input_manifest.resolve()),
        "validation_ratio_requested": validation_ratio,
        "seed": seed,
        "search_trials": search_trials,
        "total_samples": len(rows),
        "total_empty_room_groups": len(groups),
        "overall_label_counts": label_counts(rows),
        "overall_positive_rate": positive_rate(rows),
        "train": {
            "samples": len(train_rows),
            "empty_room_groups": len(train_groups),
            "label_counts": label_counts(train_rows),
            "positive_rate": positive_rate(train_rows),
            "output": str(train_output.resolve()),
        },
        "validation": {
            "samples": len(validation_rows),
            "empty_room_groups": len(validation_groups),
            "label_counts": label_counts(validation_rows),
            "positive_rate": positive_rate(validation_rows),
            "output": str(validation_output.resolve()),
        },
        "group_overlap_count": len(overlap_groups),
        "test_policy": (
            "当前数据不划分测试集；"
            "使用独立的空间布局 DPO 测试集进行外部评估"
        ),
    }

    write_json_atomic(
        report,
        report_output,
    )

    return report
