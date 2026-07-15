#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""独立生成房型维度的模型诊断报告。

该脚本不会加载模型，也不会修改训练流程。它读取：

1. 训练集 Manifest；
2. 验证集 Manifest；
3. 某个 Epoch 的 val_predictions.jsonl。

输出：

1. 验证集总体指标；
2. 各房型指标；
3. Macro / Weighted 房型 ROC-AUC；
4. RoomType Only 基线；
5. 训练集各房型标签分布。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )

from preference_reward.data.manifest import (
    PreferenceSample,
    read_preference_manifest,
)
from preference_reward.evaluation.room_type import (
    evaluate_by_room_type,
    metrics_from_predictions,
)


RAW_PROBABILITY_KEYS = (
    "p_like_raw",
    "probability_like_raw",
    "raw_probability_like",
)

CORRECTED_PROBABILITY_KEYS = (
    "p_like_prior_corrected",
    "p_like_corrected",
    "probability_like_prior_corrected",
    "corrected_probability_like",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "读取验证预测，独立计算房型维度的模型指标"
            "和 RoomType Only 基线。"
        )
    )

    parser.add_argument(
        "--train_manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/splits/train_room_type.jsonl"
        ),
        help="带 room_type 的训练集 Manifest。",
    )
    parser.add_argument(
        "--val_manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/splits/val_room_type.jsonl"
        ),
        help="带 room_type 的验证集 Manifest。",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="某个 Epoch 的 val_predictions.jsonl。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "报告输出路径。默认保存到预测文件同级目录的"
            " room_type_metrics.json。"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="分类概率阈值。",
    )
    parser.add_argument(
        "--ece_bins",
        type=int,
        default=10,
        help="ECE 分桶数量。",
    )

    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    """将输入路径解析为绝对路径。"""

    expanded_path = path.expanduser()

    if not expanded_path.is_absolute():
        expanded_path = (
            PROJECT_ROOT / expanded_path
        )

    return expanded_path.resolve()


def validate_file_exists(
    path: Path,
    description: str,
) -> None:
    """验证输入文件存在。"""

    if not path.is_file():
        raise FileNotFoundError(
            f"{description}不存在：{path}"
        )


def read_jsonl_records(
    path: Path,
) -> list[Dict[str, Any]]:
    """读取 JSONL，并校验每行均为对象。"""

    validate_file_exists(
        path,
        "JSONL 文件",
    )

    records: list[Dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON 解析失败："
                    f"{path}:{line_number}"
                ) from exc

            if not isinstance(value, dict):
                raise TypeError(
                    f"{path}:{line_number} "
                    "必须是 JSON 对象"
                )

            records.append(value)

    if not records:
        raise RuntimeError(
            f"JSONL 中没有有效记录：{path}"
        )

    return records


def find_first_value(
    record: Mapping[str, Any],
    candidate_keys: Sequence[str],
) -> Any | None:
    """返回记录中第一个存在且非空的候选字段值。"""

    for key in candidate_keys:
        if key not in record:
            continue

        value = record[key]

        if value is None:
            continue

        if (
            isinstance(value, str)
            and not value.strip()
        ):
            continue

        return value

    return None


def require_numeric_field(
    record: Mapping[str, Any],
    key: str,
    line_number: int,
) -> float:
    """读取必需的数值字段。"""

    if key not in record:
        raise KeyError(
            f"预测第 {line_number} 行缺少字段："
            f"{key}"
        )

    try:
        return float(record[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"预测第 {line_number} 行字段 "
            f"{key} 不是有效数值："
            f"{record[key]!r}"
        ) from exc


def normalize_probability(
    value: Any,
    field_name: str,
    line_number: int,
) -> float:
    """将概率字段转换为 [0, 1] 浮点数。"""

    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"预测第 {line_number} 行 "
            f"{field_name} 不是有效数值："
            f"{value!r}"
        ) from exc

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"预测第 {line_number} 行 "
            f"{field_name} 超出 [0, 1]："
            f"{probability}"
        )

    return probability


def normalize_prediction_record(
    record: Mapping[str, Any],
    line_number: int,
) -> Dict[str, Any]:
    """将不同版本的预测字段规范化为统一格式。"""

    normalized = dict(record)

    if "label" not in record:
        raise KeyError(
            f"预测第 {line_number} 行缺少 label"
        )

    try:
        label = int(record["label"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"预测第 {line_number} 行 label "
            f"不是有效整数：{record['label']!r}"
        ) from exc

    if label not in (0, 1):
        raise ValueError(
            f"预测第 {line_number} 行 label "
            f"必须是 0 或 1：{label}"
        )

    raw_probability_value = find_first_value(
        record,
        RAW_PROBABILITY_KEYS,
    )

    if raw_probability_value is None:
        raise KeyError(
            f"预测第 {line_number} 行缺少原始点赞概率。"
            f"支持字段：{RAW_PROBABILITY_KEYS}"
        )

    corrected_probability_value = (
        find_first_value(
            record,
            CORRECTED_PROBABILITY_KEYS,
        )
    )

    raw_probability = normalize_probability(
        raw_probability_value,
        "p_like_raw",
        line_number,
    )

    # 旧预测没有先验修正概率时，回退到原始概率。
    if corrected_probability_value is None:
        corrected_probability = (
            raw_probability
        )
    else:
        corrected_probability = (
            normalize_probability(
                corrected_probability_value,
                "p_like_prior_corrected",
                line_number,
            )
        )

    normalized["label"] = label
    normalized["reward_score"] = (
        require_numeric_field(
            record,
            "reward_score",
            line_number,
        )
    )
    normalized["p_like_raw"] = raw_probability
    normalized["p_like_prior_corrected"] = (
        corrected_probability
    )

    sample_id = str(
        record.get("sample_id") or ""
    ).strip()

    if sample_id:
        normalized["sample_id"] = sample_id

    return normalized


def normalize_prediction_records(
    records: Iterable[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """规范化全部逐样本预测。"""

    return [
        normalize_prediction_record(
            record=record,
            line_number=line_number,
        )
        for line_number, record in enumerate(
            records,
            start=1,
        )
    ]


def load_manifest_samples(
    path: Path,
) -> list[PreferenceSample]:
    """读取 Manifest；诊断阶段不重新检查图片文件。"""

    validate_file_exists(
        path,
        "Manifest",
    )

    return read_preference_manifest(
        manifest_path=path,
        project_root=PROJECT_ROOT,
        validate_image_paths=False,
    )


def build_output_path(
    predictions_path: Path,
    requested_output: Path | None,
) -> Path:
    """确定报告输出路径。"""

    if requested_output is None:
        return (
            predictions_path.parent
            / "room_type_metrics.json"
        )

    output_path = requested_output.expanduser()

    if not output_path.is_absolute():
        output_path = (
            PROJECT_ROOT / output_path
        )

    return output_path.resolve()


def write_json_atomic(
    value: Mapping[str, Any],
    output_path: Path,
) -> None:
    """原子写入 JSON 报告。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)

        json.dump(
            dict(value),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    os.replace(
        temporary_path,
        output_path,
    )


def build_diagnostic_report(
    train_manifest_path: Path,
    val_manifest_path: Path,
    predictions_path: Path,
    threshold: float,
    ece_bins: int,
) -> Dict[str, Any]:
    """加载输入并构建完整房型诊断报告。"""

    train_samples = load_manifest_samples(
        train_manifest_path
    )
    val_samples = load_manifest_samples(
        val_manifest_path
    )

    raw_predictions = read_jsonl_records(
        predictions_path
    )
    predictions = (
        normalize_prediction_records(
            raw_predictions
        )
    )

    model_overall = metrics_from_predictions(
        predictions=predictions,
        threshold=threshold,
        ece_bins=ece_bins,
    )

    room_type_report = evaluate_by_room_type(
        train_samples=train_samples,
        val_samples=val_samples,
        predictions=predictions,
        threshold=threshold,
        ece_bins=ece_bins,
    )

    return {
        "sources": {
            "train_manifest": str(
                train_manifest_path
            ),
            "val_manifest": str(
                val_manifest_path
            ),
            "predictions": str(
                predictions_path
            ),
        },
        "configuration": {
            "threshold": threshold,
            "ece_bins": ece_bins,
        },
        "model_overall": model_overall,
        **room_type_report,
    }


def format_metric(
    value: Any,
    decimal_places: int = 4,
) -> str:
    """将可选数值格式化为便于阅读的字符串。"""

    if value is None:
        return "N/A"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    return f"{numeric_value:.{decimal_places}f}"


def print_overall_summary(
    report: Mapping[str, Any],
) -> None:
    """打印总体指标摘要。"""

    metrics = report["model_overall"]

    print("========== MODEL OVERALL ==========")
    print(
        "samples:",
        metrics.get("num_samples"),
    )
    print(
        "roc_auc:",
        format_metric(
            metrics.get("roc_auc")
        ),
    )
    print(
        "accuracy:",
        format_metric(
            metrics.get("accuracy")
        ),
    )
    print(
        "balanced_accuracy:",
        format_metric(
            metrics.get(
                "balanced_accuracy"
            )
        ),
    )


def print_room_type_summary(
    report: Mapping[str, Any],
) -> None:
    """打印各房型指标摘要。"""

    room_metrics = report[
        "model_by_room_type"
    ]

    print()
    print("========== MODEL BY ROOM TYPE ==========")

    for room_type in sorted(room_metrics):
        metrics = room_metrics[room_type]

        print(
            f"{room_type}: "
            f"n={metrics.get('num_samples')}, "
            f"like={metrics.get('like')}, "
            f"dislike={metrics.get('dislike')}, "
            f"auc={format_metric(metrics.get('roc_auc'))}, "
            f"balanced_acc="
            f"{format_metric(metrics.get('balanced_accuracy'))}"
        )

    summary = report[
        "model_by_room_type_summary"
    ]

    print()
    print(
        "macro_room_type_auc:",
        format_metric(
            summary.get("macro_roc_auc")
        ),
    )
    print(
        "weighted_room_type_auc:",
        format_metric(
            summary.get(
                "weighted_roc_auc"
            )
        ),
    )


def print_room_type_only_summary(
    report: Mapping[str, Any],
) -> None:
    """打印 RoomType Only 基线摘要。"""

    baseline = report[
        "room_type_only_baseline"
    ]

    print()
    print("========== ROOM TYPE ONLY ==========")
    print(
        "roc_auc:",
        format_metric(
            baseline.get("roc_auc")
        ),
    )
    print(
        "accuracy:",
        format_metric(
            baseline.get("accuracy")
        ),
    )
    print(
        "balanced_accuracy:",
        format_metric(
            baseline.get(
                "balanced_accuracy"
            )
        ),
    )


def print_report_summary(
    report: Mapping[str, Any],
    output_path: Path,
) -> None:
    """打印完整的人类可读摘要。"""

    print_overall_summary(report)
    print_room_type_summary(report)
    print_room_type_only_summary(report)

    print()
    print("Report saved to:")
    print(output_path)


def validate_arguments(
    threshold: float,
    ece_bins: int,
) -> None:
    """校验数值参数。"""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold 必须位于 [0, 1]"
        )

    if ece_bins <= 0:
        raise ValueError(
            "ece_bins 必须大于 0"
        )


def main() -> None:
    """执行独立房型评估。"""

    args = parse_args()

    validate_arguments(
        threshold=args.threshold,
        ece_bins=args.ece_bins,
    )

    train_manifest_path = (
        resolve_input_path(
            args.train_manifest
        )
    )
    val_manifest_path = (
        resolve_input_path(
            args.val_manifest
        )
    )
    predictions_path = resolve_input_path(
        args.predictions
    )

    output_path = build_output_path(
        predictions_path=predictions_path,
        requested_output=args.output,
    )

    report = build_diagnostic_report(
        train_manifest_path=(
            train_manifest_path
        ),
        val_manifest_path=(
            val_manifest_path
        ),
        predictions_path=predictions_path,
        threshold=args.threshold,
        ece_bins=args.ece_bins,
    )

    write_json_atomic(
        report,
        output_path,
    )

    print_report_summary(
        report,
        output_path,
    )


if __name__ == "__main__":
    main()
