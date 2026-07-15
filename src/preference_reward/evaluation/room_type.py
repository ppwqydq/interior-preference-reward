#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""按房型评估偏好模型，并计算 RoomType Only 基线。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from preference_reward.data.manifest import (
    PreferenceSample,
)
from preference_reward.evaluation.classification import (
    classification_metrics,
)


UNKNOWN_VALUES = {
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
}


def normalize_room_type(value: object) -> str:
    """规范化房型名称。"""

    text = str(value or "").strip()

    if text.lower() in UNKNOWN_VALUES:
        return "Unknown"

    return text


def probability_to_logit(
    probability: float,
) -> float:
    """把概率转换为有限 Logit。"""

    epsilon = 1.0e-6
    clipped = min(
        max(float(probability), epsilon),
        1.0 - epsilon,
    )

    return float(
        math.log(
            clipped / (1.0 - clipped)
        )
    )


def validate_prediction_alignment(
    samples: Sequence[PreferenceSample],
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    """检查验证样本与逐样本预测是否严格对齐。"""

    if len(samples) != len(predictions):
        raise ValueError(
            "验证样本与预测数量不一致："
            f"{len(samples)} != {len(predictions)}"
        )

    for index, (sample, prediction) in enumerate(
        zip(samples, predictions),
        start=1,
    ):
        prediction_label = int(
            prediction["label"]
        )

        if prediction_label != sample.label:
            raise ValueError(
                f"第 {index} 条标签不一致："
                f"{sample.label} != "
                f"{prediction_label}"
            )

        prediction_sample_id = str(
            prediction.get("sample_id") or ""
        ).strip()

        if (
            prediction_sample_id
            and prediction_sample_id
            != sample.sample_id
        ):
            raise ValueError(
                f"第 {index} 条 sample_id 不一致："
                f"{sample.sample_id} != "
                f"{prediction_sample_id}"
            )


def metrics_from_predictions(
    predictions: Sequence[Mapping[str, Any]],
    threshold: float,
    ece_bins: int,
) -> Dict[str, Any]:
    """使用现有逐样本预测计算分类指标。"""

    return classification_metrics(
        labels_list=[
            int(row["label"])
            for row in predictions
        ],
        reward_scores_list=[
            float(row["reward_score"])
            for row in predictions
        ],
        raw_probabilities_list=[
            float(row["p_like_raw"])
            for row in predictions
        ],
        corrected_probabilities_list=[
            float(
                row["p_like_prior_corrected"]
            )
            for row in predictions
        ],
        threshold=threshold,
        ece_bins=ece_bins,
    )


def build_training_room_type_distribution(
    train_samples: Sequence[PreferenceSample],
) -> tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, float],
    float,
]:
    """统计训练集房型分布及房型点赞率。"""

    if not train_samples:
        raise ValueError("训练集不能为空")

    labels_by_room: Dict[
        str,
        List[int],
    ] = defaultdict(list)

    all_labels: List[int] = []

    for sample in train_samples:
        room_type = normalize_room_type(
            getattr(
                sample,
                "room_type",
                "Unknown",
            )
        )

        label = int(sample.label)

        labels_by_room[room_type].append(
            label
        )
        all_labels.append(label)

    global_like_rate = float(
        sum(all_labels) / len(all_labels)
    )

    distribution: Dict[
        str,
        Dict[str, Any],
    ] = {}
    like_rates: Dict[str, float] = {}

    for room_type in sorted(labels_by_room):
        labels = labels_by_room[room_type]

        likes = int(sum(labels))
        total = len(labels)
        dislikes = total - likes
        like_rate = float(likes / total)

        distribution[room_type] = {
            "samples": total,
            "like": likes,
            "dislike": dislikes,
            "like_rate": like_rate,
        }
        like_rates[room_type] = like_rate

    return (
        distribution,
        like_rates,
        global_like_rate,
    )


def build_room_type_only_baseline(
    val_samples: Sequence[PreferenceSample],
    train_room_like_rates: Mapping[str, float],
    global_train_like_rate: float,
    threshold: float,
    ece_bins: int,
) -> Dict[str, Any]:
    """仅使用训练集房型点赞率预测验证集。"""

    labels: List[int] = []
    probabilities: List[float] = []
    reward_scores: List[float] = []

    for sample in val_samples:
        room_type = normalize_room_type(
            getattr(
                sample,
                "room_type",
                "Unknown",
            )
        )

        probability = float(
            train_room_like_rates.get(
                room_type,
                global_train_like_rate,
            )
        )

        labels.append(int(sample.label))
        probabilities.append(probability)
        reward_scores.append(
            probability_to_logit(
                probability
            )
        )

    metrics = classification_metrics(
        labels_list=labels,
        reward_scores_list=reward_scores,
        raw_probabilities_list=probabilities,
        corrected_probabilities_list=(
            probabilities
        ),
        threshold=threshold,
        ece_bins=ece_bins,
    )

    metrics["score_definition"] = (
        "训练集中对应 room_type 的 like_rate；"
        "未见房型回退为训练集总体 like_rate"
    )
    metrics["global_train_like_rate"] = (
        global_train_like_rate
    )

    return metrics


def summarize_room_type_auc(
    room_metrics: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> Dict[str, Any]:
    """计算房型 AUC 的 Macro 和加权汇总。"""

    values = []

    for room_type, metrics in (
        room_metrics.items()
    ):
        auc = metrics.get("roc_auc")
        sample_count = int(
            metrics["num_samples"]
        )

        if auc is None:
            continue

        numeric_auc = float(auc)

        if not math.isfinite(numeric_auc):
            continue

        values.append(
            (
                room_type,
                numeric_auc,
                sample_count,
            )
        )

    if not values:
        return {
            "valid_room_types": 0,
            "macro_roc_auc": None,
            "weighted_roc_auc": None,
        }

    return {
        "valid_room_types": len(values),
        "macro_roc_auc": float(
            np.mean(
                [
                    auc
                    for _, auc, _
                    in values
                ]
            )
        ),
        "weighted_roc_auc": float(
            np.average(
                [
                    auc
                    for _, auc, _
                    in values
                ],
                weights=[
                    count
                    for _, _, count
                    in values
                ],
            )
        ),
        "included_room_types": [
            room_type
            for room_type, _, _
            in values
        ],
    }


def evaluate_by_room_type(
    train_samples: Sequence[PreferenceSample],
    val_samples: Sequence[PreferenceSample],
    predictions: Sequence[Mapping[str, Any]],
    threshold: float,
    ece_bins: int,
) -> Dict[str, Any]:
    """生成完整房型诊断报告。"""

    validate_prediction_alignment(
        samples=val_samples,
        predictions=predictions,
    )

    grouped_predictions: Dict[
        str,
        List[Mapping[str, Any]],
    ] = defaultdict(list)

    val_room_label_counts = Counter()

    for sample, prediction in zip(
        val_samples,
        predictions,
    ):
        room_type = normalize_room_type(
            getattr(
                sample,
                "room_type",
                "Unknown",
            )
        )

        grouped_predictions[
            room_type
        ].append(prediction)

        val_room_label_counts[
            (
                room_type,
                int(sample.label),
            )
        ] += 1

    model_by_room_type: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for room_type in sorted(
        grouped_predictions
    ):
        room_predictions = (
            grouped_predictions[room_type]
        )

        metrics = metrics_from_predictions(
            predictions=room_predictions,
            threshold=threshold,
            ece_bins=ece_bins,
        )

        metrics["like"] = int(
            val_room_label_counts[
                (room_type, 1)
            ]
        )
        metrics["dislike"] = int(
            val_room_label_counts[
                (room_type, 0)
            ]
        )

        model_by_room_type[
            room_type
        ] = metrics

    (
        train_distribution,
        train_room_like_rates,
        global_train_like_rate,
    ) = build_training_room_type_distribution(
        train_samples
    )

    room_type_only = (
        build_room_type_only_baseline(
            val_samples=val_samples,
            train_room_like_rates=(
                train_room_like_rates
            ),
            global_train_like_rate=(
                global_train_like_rate
            ),
            threshold=threshold,
            ece_bins=ece_bins,
        )
    )

    return {
        "model_by_room_type": (
            model_by_room_type
        ),
        "model_by_room_type_summary": (
            summarize_room_type_auc(
                model_by_room_type
            )
        ),
        "room_type_only_baseline": (
            room_type_only
        ),
        "train_room_type_distribution": (
            train_distribution
        ),
        "policy": {
            "checkpoint_selection_metric": (
                "overall validation roc_auc"
            ),
            "room_type_metrics_usage": (
                "diagnostic_only"
            ),
            "unknown_room_type_policy": (
                "group as Unknown; prompt omitted"
            ),
        },
    }
