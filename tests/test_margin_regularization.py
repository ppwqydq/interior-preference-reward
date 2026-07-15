#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest
import torch

from preference_reward.training.margin_regularization import (
    build_margin_regularized_loss,
    mean_squared_reward_margin,
    reward_margins,
    validate_margin_l2_weight,
)


def test_reward_margins() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ]
    )

    margins = reward_margins(logits)

    assert torch.allclose(
        margins,
        torch.tensor([2.0, -1.0]),
    )


def test_mean_squared_reward_margin() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ]
    )

    penalty = (
        mean_squared_reward_margin(
            logits
        )
    )

    # (2^2 + (-1)^2) / 2 = 2.5
    assert penalty.item() == pytest.approx(
        2.5
    )


def test_combined_loss() -> None:
    classification_loss = torch.tensor(
        0.75
    )
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        requires_grad=True,
    )

    result = build_margin_regularized_loss(
        classification_loss=(
            classification_loss
        ),
        ab_logits=logits,
        margin_l2_weight=0.1,
    )

    assert (
        result.margin_penalty.item()
        == pytest.approx(2.5)
    )
    assert (
        result.weighted_margin_penalty.item()
        == pytest.approx(0.25)
    )
    assert (
        result.total_loss.item()
        == pytest.approx(1.0)
    )

    result.total_loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()


def test_zero_weight_preserves_loss() -> None:
    classification_loss = torch.tensor(
        0.75,
        requires_grad=True,
    )
    logits = torch.tensor(
        [[100.0, -100.0]],
        requires_grad=True,
    )

    result = build_margin_regularized_loss(
        classification_loss=(
            classification_loss
        ),
        ab_logits=logits,
        margin_l2_weight=0.0,
    )

    assert (
        result.total_loss
        is classification_loss
    )
    assert (
        result.margin_penalty.item()
        == 0.0
    )
    assert (
        result.weighted_margin_penalty.item()
        == 0.0
    )


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="不能小于 0",
    ):
        validate_margin_l2_weight(-0.01)


def test_invalid_logit_shape_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="第二维必须为 2",
    ):
        reward_margins(
            torch.zeros(2, 3)
        )
