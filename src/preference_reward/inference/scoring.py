#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""通用双图 Reward Model 推理评分。

输入：

    原始房间图 + 候选生成图

输出：

    A/B logits
    reward_score
    p_like_raw
    p_like_prior_corrected

本模块只负责推理评分：

- 不读取训练集；
- 不计算训练损失；
- 不更新模型参数；
- 不包含 GOOD/BAD 或具体数据集逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from preference_reward.models.qwen_ab_reward import (
    QwenABRewardBackend,
)


@dataclass(frozen=True)
class RewardScoringSample:
    """一条通用双图评分输入。

    字段名称与 QwenABRewardBackend 的双图输入约定一致：

        empty_room_image
            原始房间或参考图。

        generated_furniture_image
            待评分的候选生成图。
    """

    sample_id: str
    empty_room_image: Path
    generated_furniture_image: Path
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


def batched_scoring_samples(
    samples: Sequence[RewardScoringSample],
    batch_size: int,
) -> Iterable[Sequence[RewardScoringSample]]:
    """按指定大小切分推理 Batch。"""

    if batch_size <= 0:
        raise ValueError(
            "batch_size 必须大于 0"
        )

    for start in range(
        0,
        len(samples),
        batch_size,
    ):
        yield samples[
            start:start + batch_size
        ]


def validate_scoring_samples(
    samples: Sequence[RewardScoringSample],
) -> None:
    """校验评分输入。"""

    if not samples:
        raise ValueError(
            "评分样本不能为空"
        )

    seen_sample_ids: set[str] = set()
    missing_paths: List[str] = []

    for sample in samples:
        if not sample.sample_id:
            raise ValueError(
                "sample_id 不能为空"
            )

        if sample.sample_id in seen_sample_ids:
            raise ValueError(
                f"sample_id 重复："
                f"{sample.sample_id}"
            )

        seen_sample_ids.add(
            sample.sample_id
        )

        if not sample.empty_room_image.is_file():
            missing_paths.append(
                "sample_id="
                f"{sample.sample_id} "
                "role=reference "
                f"path={sample.empty_room_image}"
            )

        if (
            not sample
            .generated_furniture_image
            .is_file()
        ):
            missing_paths.append(
                "sample_id="
                f"{sample.sample_id} "
                "role=candidate "
                "path="
                f"{sample.generated_furniture_image}"
            )

    if missing_paths:
        preview = "\n".join(
            f"  - {item}"
            for item in missing_paths[:20]
        )

        raise FileNotFoundError(
            f"共有 {len(missing_paths)} 个"
            "推理图片路径不存在：\n"
            f"{preview}"
        )


@torch.inference_mode()
def score_reward_samples(
    backend: QwenABRewardBackend,
    samples: Sequence[RewardScoringSample],
    batch_size: int,
    negative_weight: float | None = None,
) -> List[Dict[str, Any]]:
    """批量计算双图 Reward 分数。

    reward_score 定义：

        logit_A_like - logit_B_dislike

    排序和 Pairwise 比较应直接使用 reward_score。

    p_like_raw 是 A/B 原始二分类倾向。

    p_like_prior_corrected 仅在传入 negative_weight 时计算，
    用于诊断类别加权带来的先验变化，不作为主排序分数。
    """

    validate_scoring_samples(
        samples
    )

    if (
        negative_weight is not None
        and negative_weight <= 0
    ):
        raise ValueError(
            "negative_weight 必须大于 0"
        )

    backend.model.eval()

    results: List[Dict[str, Any]] = []

    for batch_samples in batched_scoring_samples(
        samples=samples,
        batch_size=batch_size,
    ):
        # RewardScoringSample 提供了 Backend 所需的两个图像字段。
        inputs = backend.make_inputs(
            batch_samples
        )

        ab_logits, last_positions = (
            backend.forward_ab_logits(
                inputs
            )
        )

        reward_scores = (
            backend.reward_scores(
                ab_logits
            )
        )

        raw_probabilities = (
            backend.raw_probabilities(
                ab_logits
            )
        )

        corrected_probabilities = None

        if negative_weight is not None:
            corrected_probabilities = (
                backend
                .prior_corrected_probabilities(
                    ab_logits,
                    negative_weight,
                )
            )

        logits_cpu = (
            ab_logits
            .detach()
            .cpu()
            .tolist()
        )

        rewards_cpu = (
            reward_scores
            .detach()
            .cpu()
            .tolist()
        )

        raw_probabilities_cpu = (
            raw_probabilities
            .detach()
            .cpu()
            .tolist()
        )

        positions_cpu = (
            last_positions
            .detach()
            .cpu()
            .tolist()
        )

        if corrected_probabilities is None:
            corrected_probabilities_cpu = [
                None
                for _ in batch_samples
            ]
        else:
            corrected_probabilities_cpu = (
                corrected_probabilities
                .detach()
                .cpu()
                .tolist()
            )

        for (
            sample,
            logits,
            reward_score,
            p_like_raw,
            p_like_prior_corrected,
            last_position,
        ) in zip(
            batch_samples,
            logits_cpu,
            rewards_cpu,
            raw_probabilities_cpu,
            corrected_probabilities_cpu,
            positions_cpu,
        ):
            result: Dict[str, Any] = {
                "sample_id": (
                    sample.sample_id
                ),
                "reference_image_path": str(
                    sample.empty_room_image
                ),
                "candidate_image_path": str(
                    sample
                    .generated_furniture_image
                ),
                "logit_A_like": float(
                    logits[0]
                ),
                "logit_B_dislike": float(
                    logits[1]
                ),
                "reward_score": float(
                    reward_score
                ),
                "p_like_raw": float(
                    p_like_raw
                ),
                "last_valid_position": int(
                    last_position
                ),
                "metadata": dict(
                    sample.metadata
                ),
            }

            if (
                p_like_prior_corrected
                is not None
            ):
                result[
                    "p_like_prior_corrected"
                ] = float(
                    p_like_prior_corrected
                )

            results.append(result)

        del inputs
        del ab_logits
        del last_positions
        del reward_scores
        del raw_probabilities

        if corrected_probabilities is not None:
            del corrected_probabilities

    return results
