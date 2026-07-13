#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pointwise 偏好分类指标与验证逻辑。

主要排序分数：

    reward_score = logit_A - logit_B

概率分为：

    p_like_raw
        A/B 原始 softmax 分数，受类别加权影响。

    p_like_prior_corrected
        根据负类损失权重进行理论先验修正后的概率。

正式生产概率仍建议在独立校准集上进行 Platt Calibration。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

from preference_reward.data.manifest import (
    PreferenceSample,
    batched,
)
from preference_reward.models.qwen_ab_reward import (
    QwenABRewardBackend,
)


def safe_roc_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float | None:
    """安全计算 ROC-AUC。"""

    try:
        return float(
            roc_auc_score(labels, scores)
        )
    except ValueError:
        return None


def safe_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float | None:
    """安全计算 Average Precision。"""

    try:
        return float(
            average_precision_score(
                labels,
                scores,
            )
        )
    except ValueError:
        return None


def compute_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> float:
    """计算 Expected Calibration Error。"""

    if bins <= 0:
        raise ValueError(
            "ECE bins 必须大于 0"
        )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )
    ece = 0.0

    for index in range(bins):
        left = edges[index]
        right = edges[index + 1]

        if index == bins - 1:
            mask = (
                (probabilities >= left)
                & (probabilities <= right)
            )
        else:
            mask = (
                (probabilities >= left)
                & (probabilities < right)
            )

        if not np.any(mask):
            continue

        confidence = float(
            probabilities[mask].mean()
        )
        observed_rate = float(
            labels[mask].mean()
        )

        ece += (
            float(mask.mean())
            * abs(
                confidence
                - observed_rate
            )
        )

    return float(ece)


def optional_mean(
    values: np.ndarray,
) -> float | None:
    """空数组返回 None，否则返回均值。"""

    if values.size == 0:
        return None

    return float(values.mean())


def difference(
    positive_value: float | None,
    negative_value: float | None,
) -> float | None:
    """安全计算正负样本均值差。"""

    if (
        positive_value is None
        or negative_value is None
    ):
        return None

    return float(
        positive_value - negative_value
    )


def classification_metrics(
    labels_list: Sequence[int],
    reward_scores_list: Sequence[float],
    raw_probabilities_list: Sequence[float],
    corrected_probabilities_list: Sequence[float],
    threshold: float,
    ece_bins: int,
) -> Dict[str, Any]:
    """计算排序、分类和概率指标。"""

    labels = np.asarray(
        labels_list,
        dtype=np.int64,
    )
    reward_scores = np.asarray(
        reward_scores_list,
        dtype=np.float64,
    )
    raw_probabilities = np.asarray(
        raw_probabilities_list,
        dtype=np.float64,
    )
    corrected_probabilities = np.asarray(
        corrected_probabilities_list,
        dtype=np.float64,
    )

    if not (
        len(labels)
        == len(reward_scores)
        == len(raw_probabilities)
        == len(corrected_probabilities)
    ):
        raise ValueError(
            "指标输入长度不一致"
        )

    corrected_predictions = (
        corrected_probabilities
        >= threshold
    ).astype(np.int64)

    raw_predictions = (
        raw_probabilities
        >= threshold
    ).astype(np.int64)

    positive_mask = labels == 1
    negative_mask = labels == 0

    mean_reward_positive = optional_mean(
        reward_scores[positive_mask]
    )
    mean_reward_negative = optional_mean(
        reward_scores[negative_mask]
    )

    mean_raw_positive = optional_mean(
        raw_probabilities[positive_mask]
    )
    mean_raw_negative = optional_mean(
        raw_probabilities[negative_mask]
    )

    mean_corrected_positive = optional_mean(
        corrected_probabilities[
            positive_mask
        ]
    )
    mean_corrected_negative = optional_mean(
        corrected_probabilities[
            negative_mask
        ]
    )

    raw_brier = float(
        brier_score_loss(
            labels,
            raw_probabilities,
        )
    )
    raw_ece = compute_ece(
        labels,
        raw_probabilities,
        ece_bins,
    )

    corrected_brier = float(
        brier_score_loss(
            labels,
            corrected_probabilities,
        )
    )
    corrected_ece = compute_ece(
        labels,
        corrected_probabilities,
        ece_bins,
    )

    reward_difference = difference(
        mean_reward_positive,
        mean_reward_negative,
    )

    return {
        "num_samples": int(len(labels)),
        "threshold": float(threshold),

        # 主要排序指标直接使用 reward_score。
        "roc_auc": safe_roc_auc(
            labels,
            reward_scores,
        ),
        "pr_auc_positive": (
            safe_average_precision(
                labels,
                reward_scores,
            )
        ),
        "pr_auc_negative": (
            safe_average_precision(
                1 - labels,
                -reward_scores,
            )
        ),

        # 默认分类指标使用先验修正后的概率。
        "accuracy": float(
            accuracy_score(
                labels,
                corrected_predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                corrected_predictions,
            )
        ),
        "confusion_matrix_labels_0_1": (
            confusion_matrix(
                labels,
                corrected_predictions,
                labels=[0, 1],
            ).tolist()
        ),
        "positive_recall": float(
            recall_score(
                labels,
                corrected_predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "negative_recall": float(
            recall_score(
                labels,
                corrected_predictions,
                pos_label=0,
                zero_division=0,
            )
        ),

        # 原始 A/B 分数下的分类结果，仅供诊断。
        "raw_accuracy_at_threshold": float(
            accuracy_score(
                labels,
                raw_predictions,
            )
        ),
        "raw_balanced_accuracy_at_threshold": float(
            balanced_accuracy_score(
                labels,
                raw_predictions,
            )
        ),

        # 概率质量。
        "raw_brier": raw_brier,
        "raw_ece": raw_ece,
        "prior_corrected_brier": (
            corrected_brier
        ),
        "prior_corrected_ece": (
            corrected_ece
        ),

        # 保留训练器原有字段名。
        # 当前指向先验修正后的概率指标。
        "brier": corrected_brier,
        "ece": corrected_ece,

        # Reward score 分布。
        "mean_reward_score_positive": (
            mean_reward_positive
        ),
        "mean_reward_score_negative": (
            mean_reward_negative
        ),
        "mean_reward_score_positive_minus_negative": (
            reward_difference
        ),

        # 兼容原训练日志中的 margin 字段。
        "mean_margin_positive_minus_negative": (
            reward_difference
        ),

        # 原始 A/B 点赞分数。
        "mean_p_like_raw_positive": (
            mean_raw_positive
        ),
        "mean_p_like_raw_negative": (
            mean_raw_negative
        ),
        "mean_p_like_raw_positive_minus_negative": (
            difference(
                mean_raw_positive,
                mean_raw_negative,
            )
        ),

        # 先验修正后的点赞概率。
        "mean_p_like_prior_corrected_positive": (
            mean_corrected_positive
        ),
        "mean_p_like_prior_corrected_negative": (
            mean_corrected_negative
        ),
        "mean_p_like_prior_corrected_positive_minus_negative": (
            difference(
                mean_corrected_positive,
                mean_corrected_negative,
            )
        ),

        "predicted_like_rate_raw": float(
            raw_predictions.mean()
        ),
        "predicted_like_rate_prior_corrected": float(
            corrected_predictions.mean()
        ),

        # 分数离散程度诊断。
        "reward_score_unique_count": int(
            np.unique(reward_scores).size
        ),
        "reward_score_std": float(
            reward_scores.std()
        ),
        "p_like_raw_std": float(
            raw_probabilities.std()
        ),
    }


@torch.inference_mode()
def evaluate_samples(
    backend: QwenABRewardBackend,
    samples: Sequence[PreferenceSample],
    batch_size: int,
    negative_weight: float,
    threshold: float,
    ece_bins: int,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """执行验证并输出逐样本预测。"""

    backend.model.eval()

    labels_all: List[int] = []
    reward_scores_all: List[float] = []
    raw_probabilities_all: List[float] = []
    corrected_probabilities_all: List[float] = []
    predictions: List[Dict[str, Any]] = []

    weighted_loss_sum = 0.0
    unweighted_loss_sum = 0.0
    sample_count = 0

    for batch_samples in batched(
        samples,
        batch_size,
    ):
        inputs = backend.make_inputs(
            batch_samples
        )
        targets = backend.make_targets(
            batch_samples
        )

        ab_logits, positions = (
            backend.forward_ab_logits(inputs)
        )

        weighted_loss = (
            backend.weighted_cross_entropy(
                ab_logits,
                targets,
                negative_weight,
            )
        )

        unweighted_loss = (
            torch.nn.functional.cross_entropy(
                ab_logits,
                targets,
                reduction="sum",
            )
        )

        reward_scores = backend.reward_scores(
            ab_logits
        )
        raw_probabilities = (
            backend.raw_probabilities(
                ab_logits
            )
        )
        corrected_probabilities = (
            backend.prior_corrected_probabilities(
                ab_logits,
                negative_weight,
            )
        )

        actual_batch_size = len(
            batch_samples
        )

        weighted_loss_sum += (
            float(weighted_loss.item())
            * actual_batch_size
        )
        unweighted_loss_sum += float(
            unweighted_loss.item()
        )
        sample_count += actual_batch_size

        logits_cpu = (
            ab_logits.detach().cpu().tolist()
        )
        reward_scores_cpu = (
            reward_scores.detach().cpu().tolist()
        )
        raw_probabilities_cpu = (
            raw_probabilities
            .detach()
            .cpu()
            .tolist()
        )
        corrected_probabilities_cpu = (
            corrected_probabilities
            .detach()
            .cpu()
            .tolist()
        )
        positions_cpu = (
            positions.detach().cpu().tolist()
        )

        for (
            sample,
            logits_pair,
            reward_score,
            raw_probability,
            corrected_probability,
            position,
        ) in zip(
            batch_samples,
            logits_cpu,
            reward_scores_cpu,
            raw_probabilities_cpu,
            corrected_probabilities_cpu,
            positions_cpu,
        ):
            labels_all.append(sample.label)
            reward_scores_all.append(
                float(reward_score)
            )
            raw_probabilities_all.append(
                float(raw_probability)
            )
            corrected_probabilities_all.append(
                float(corrected_probability)
            )

            predictions.append(
                {
                    "sample_id": sample.sample_id,
                    "empty_room_image": str(
                        sample.empty_room_image
                    ),
                    "generated_furniture_image": str(
                        sample.generated_furniture_image
                    ),
                    "label": sample.label,

                    "logit_A_like": float(
                        logits_pair[0]
                    ),
                    "logit_B_dislike": float(
                        logits_pair[1]
                    ),
                    "reward_score": float(
                        reward_score
                    ),

                    "p_like_raw": float(
                        raw_probability
                    ),
                    "p_like_prior_corrected": float(
                        corrected_probability
                    ),

                    "prediction_raw": int(
                        raw_probability >= threshold
                    ),
                    "prediction": int(
                        corrected_probability
                        >= threshold
                    ),

                    "last_valid_position": int(
                        position
                    ),
                }
            )

        del (
            inputs,
            targets,
            ab_logits,
            positions,
            weighted_loss,
            unweighted_loss,
            reward_scores,
            raw_probabilities,
            corrected_probabilities,
        )

    metrics = classification_metrics(
        labels_list=labels_all,
        reward_scores_list=reward_scores_all,
        raw_probabilities_list=(
            raw_probabilities_all
        ),
        corrected_probabilities_list=(
            corrected_probabilities_all
        ),
        threshold=threshold,
        ece_bins=ece_bins,
    )

    metrics.update(
        {
            "validation_weighted_loss": (
                weighted_loss_sum
                / sample_count
            ),
            "validation_unweighted_ce": (
                unweighted_loss_sum
                / sample_count
            ),
        }
    )

    return metrics, predictions
