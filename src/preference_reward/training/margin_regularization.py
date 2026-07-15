#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reward Margin 正则化。

该模块只负责根据 A/B Logit 计算 Margin 正则，
不负责模型前向、分类损失或训练循环。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MarginRegularizedLoss:
    """分类损失与 Margin 正则的组合结果。"""

    total_loss: torch.Tensor
    classification_loss: torch.Tensor
    margin_penalty: torch.Tensor
    weighted_margin_penalty: torch.Tensor


def validate_margin_l2_weight(
    weight: float,
) -> float:
    """校验并返回非负 Margin 正则权重。"""

    numeric_weight = float(weight)

    if numeric_weight < 0:
        raise ValueError(
            "margin_l2_weight 不能小于 0"
        )

    return numeric_weight


def validate_ab_logits(
    ab_logits: torch.Tensor,
) -> None:
    """校验 A/B Logit 的形状。"""

    if ab_logits.ndim != 2:
        raise ValueError(
            "ab_logits 必须是二维 Tensor，"
            f"实际 shape={tuple(ab_logits.shape)}"
        )

    if ab_logits.shape[1] != 2:
        raise ValueError(
            "ab_logits 的第二维必须为 2，"
            "依次对应 A-like 和 B-dislike，"
            f"实际 shape={tuple(ab_logits.shape)}"
        )

    if ab_logits.shape[0] <= 0:
        raise ValueError(
            "ab_logits 不能是空 Batch"
        )


def reward_margins(
    ab_logits: torch.Tensor,
) -> torch.Tensor:
    """计算每条样本的 Reward Margin。

    reward_margin = logit_A_like - logit_B_dislike
    """

    validate_ab_logits(ab_logits)

    return (
        ab_logits[:, 0]
        - ab_logits[:, 1]
    )


def mean_squared_reward_margin(
    ab_logits: torch.Tensor,
) -> torch.Tensor:
    """计算 Batch 内 Reward Margin 的均方值。

    使用 float32 计算正则，避免半精度平方时
    精度不足或数值范围过小。
    """

    margins = reward_margins(
        ab_logits
    ).float()

    return margins.square().mean()


def zero_scalar_like(
    reference: torch.Tensor,
) -> torch.Tensor:
    """创建与参考 Tensor 同设备的标量零。"""

    return reference.new_zeros(())


def build_margin_regularized_loss(
    classification_loss: torch.Tensor,
    ab_logits: torch.Tensor,
    margin_l2_weight: float,
) -> MarginRegularizedLoss:
    """组合分类损失与 Reward Margin L2 正则。"""

    if classification_loss.ndim != 0:
        raise ValueError(
            "classification_loss 必须是标量 Tensor"
        )

    weight = validate_margin_l2_weight(
        margin_l2_weight
    )

    # 权重为零时直接返回原始分类损失，
    # 保持旧配置的训练行为不变。
    if weight == 0.0:
        zero = zero_scalar_like(
            classification_loss
        )

        return MarginRegularizedLoss(
            total_loss=classification_loss,
            classification_loss=(
                classification_loss
            ),
            margin_penalty=zero,
            weighted_margin_penalty=zero,
        )

    margin_penalty = (
        mean_squared_reward_margin(
            ab_logits
        )
    )
    weighted_margin_penalty = (
        margin_penalty * weight
    )
    total_loss = (
        classification_loss
        + weighted_margin_penalty
    )

    return MarginRegularizedLoss(
        total_loss=total_loss,
        classification_loss=(
            classification_loss
        ),
        margin_penalty=margin_penalty,
        weighted_margin_penalty=(
            weighted_margin_penalty
        ),
    )
