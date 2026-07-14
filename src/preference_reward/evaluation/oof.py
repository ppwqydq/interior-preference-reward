#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OOF 预测结果校验、汇总与样本可信度分析。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from preference_reward.evaluation.classification import (
    classification_metrics,
)


_REQUIRED_PREDICTION_FIELDS = {
    "sample_id",
    "fold",
    "label",
    "reward_score",
    "p_like_raw",
    "p_like_prior_corrected",
}


def _require_probability(
    value: Any,
    field_name: str,
    sample_id: str,
) -> float:
    """读取并校验概率字段。"""

    probability = float(value)

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{field_name} 不在 [0, 1]："
            f"sample_id={sample_id}，value={probability}"
        )

    return probability


def enrich_oof_predictions(
    predictions: Iterable[Mapping[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """校验预测并补充标签可信度字段。

    可信度定义：

        label = 1:
            confidence = p_like

        label = 0:
            confidence = 1 - p_like

    该值只表示折外模型对原标签的支持程度，
    不应直接解释为标签真实正确概率。
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold 必须位于 [0, 1]"
        )

    enriched: List[Dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    for source in predictions:
        row = dict(source)
        missing = (
            _REQUIRED_PREDICTION_FIELDS
            - set(row)
        )

        if missing:
            raise KeyError(
                f"OOF 预测缺少字段：{sorted(missing)}"
            )

        sample_id = str(row["sample_id"]).strip()

        if not sample_id:
            raise ValueError(
                "OOF 预测 sample_id 不能为空"
            )

        if sample_id in seen_sample_ids:
            raise RuntimeError(
                f"OOF 预测 sample_id 重复：{sample_id}"
            )

        seen_sample_ids.add(sample_id)

        fold = int(row["fold"])

        if fold <= 0:
            raise ValueError(
                f"fold 必须大于 0：sample_id={sample_id}"
            )

        label = int(row["label"])

        if label not in (0, 1):
            raise ValueError(
                f"标签必须为 0 或 1："
                f"sample_id={sample_id}，label={label}"
            )

        p_like_raw = _require_probability(
            row["p_like_raw"],
            "p_like_raw",
            sample_id,
        )
        p_like_corrected = _require_probability(
            row["p_like_prior_corrected"],
            "p_like_prior_corrected",
            sample_id,
        )

        raw_label_confidence = (
            p_like_raw
            if label == 1
            else 1.0 - p_like_raw
        )
        corrected_label_confidence = (
            p_like_corrected
            if label == 1
            else 1.0 - p_like_corrected
        )

        prediction = int(
            p_like_corrected >= threshold
        )

        row.update(
            {
                "sample_id": sample_id,
                "fold": fold,
                "label": label,
                "reward_score": float(
                    row["reward_score"]
                ),
                "p_like_raw": p_like_raw,
                "p_like_prior_corrected": (
                    p_like_corrected
                ),
                "prediction": prediction,
                "label_agreement": bool(
                    prediction == label
                ),
                "label_confidence_raw": float(
                    raw_label_confidence
                ),
                "label_confidence": float(
                    corrected_label_confidence
                ),
                "conflict_score": float(
                    1.0
                    - corrected_label_confidence
                ),
            }
        )
        enriched.append(row)

    if not enriched:
        raise ValueError(
            "OOF 预测不能为空"
        )

    return sorted(
        enriched,
        key=lambda row: (
            int(row["fold"]),
            str(row["sample_id"]),
        ),
    )


def validate_oof_coverage(
    predictions: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """校验 OOF 预测与划分清单一一对应。"""

    prediction_by_id = {
        str(row["sample_id"]): row
        for row in predictions
    }
    assignment_by_id = {
        str(row["sample_id"]): row
        for row in assignments
    }

    if len(prediction_by_id) != len(predictions):
        raise RuntimeError(
            "OOF 预测中存在重复 sample_id"
        )

    if len(assignment_by_id) != len(assignments):
        raise RuntimeError(
            "OOF assignments 中存在重复 sample_id"
        )

    prediction_ids = set(prediction_by_id)
    assignment_ids = set(assignment_by_id)

    missing = sorted(
        assignment_ids - prediction_ids
    )
    unexpected = sorted(
        prediction_ids - assignment_ids
    )

    fold_mismatches: List[str] = []
    label_mismatches: List[str] = []

    for sample_id in sorted(
        prediction_ids & assignment_ids
    ):
        prediction = prediction_by_id[sample_id]
        assignment = assignment_by_id[sample_id]

        if int(prediction["fold"]) != int(
            assignment["outer_fold"]
        ):
            fold_mismatches.append(sample_id)

        if int(prediction["label"]) != int(
            assignment["label"]
        ):
            label_mismatches.append(sample_id)

    if (
        missing
        or unexpected
        or fold_mismatches
        or label_mismatches
    ):
        raise RuntimeError(
            "OOF 预测覆盖校验失败："
            f"missing={missing[:10]}，"
            f"unexpected={unexpected[:10]}，"
            f"fold_mismatches={fold_mismatches[:10]}，"
            f"label_mismatches={label_mismatches[:10]}"
        )

    fold_counts = Counter(
        int(row["fold"])
        for row in predictions
    )

    return {
        "expected_samples": len(assignments),
        "predicted_samples": len(predictions),
        "missing_samples": 0,
        "unexpected_samples": 0,
        "fold_mismatches": 0,
        "label_mismatches": 0,
        "fold_counts": {
            str(key): int(value)
            for key, value
            in sorted(fold_counts.items())
        },
    }


def compute_oof_metrics(
    predictions: Sequence[Mapping[str, Any]],
    threshold: float,
    ece_bins: int,
) -> Dict[str, Any]:
    """复用项目现有分类指标计算 OOF 指标。"""

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


def build_confidence_report(
    predictions: Sequence[Mapping[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    """生成 OOF 标签可信度与强冲突样本报告。"""

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0"
        )

    confidences = [
        float(row["label_confidence"])
        for row in predictions
    ]

    thresholds = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]

    confidence_counts = {
        f"confidence_le_{threshold:.1f}": sum(
            confidence <= threshold
            for confidence in confidences
        )
        for threshold in thresholds
    }

    sorted_conflicts = sorted(
        predictions,
        key=lambda row: (
            float(row["label_confidence"]),
            str(row["sample_id"]),
        ),
    )

    top_conflicts: List[Dict[str, Any]] = []

    for row in sorted_conflicts[:top_k]:
        top_conflicts.append(
            {
                "sample_id": str(
                    row["sample_id"]
                ),
                "fold": int(row["fold"]),
                "label": int(row["label"]),
                "prediction": int(
                    row["prediction"]
                ),
                "label_agreement": bool(
                    row["label_agreement"]
                ),
                "label_confidence": float(
                    row["label_confidence"]
                ),
                "conflict_score": float(
                    row["conflict_score"]
                ),
                "reward_score": float(
                    row["reward_score"]
                ),
                "p_like_raw": float(
                    row["p_like_raw"]
                ),
                "p_like_prior_corrected": float(
                    row[
                        "p_like_prior_corrected"
                    ]
                ),
                "empty_room_image": str(
                    row.get(
                        "empty_room_image",
                        row.get(
                            "reference_image_path",
                            "",
                        ),
                    )
                ),
                "generated_furniture_image": str(
                    row.get(
                        "generated_furniture_image",
                        row.get(
                            "candidate_image_path",
                            "",
                        ),
                    )
                ),
            }
        )

    agreement_count = sum(
        bool(row["label_agreement"])
        for row in predictions
    )

    mean_confidence = (
        sum(confidences) / len(confidences)
    )

    return {
        "num_samples": len(predictions),
        "mean_label_confidence": float(
            mean_confidence
        ),
        "agreement_count": int(
            agreement_count
        ),
        "disagreement_count": int(
            len(predictions) - agreement_count
        ),
        "agreement_rate": float(
            agreement_count / len(predictions)
        ),
        **confidence_counts,
        "top_k": min(
            top_k,
            len(predictions),
        ),
        "top_conflicts": top_conflicts,
        "interpretation": (
            "label_confidence 是折外模型对原始用户标签的支持程度，"
            "不是标签真实正确概率；低可信样本需要人工复核，"
            "不应自动删除或改标。"
        ),
    }
