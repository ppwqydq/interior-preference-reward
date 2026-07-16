#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from preference_reward.models.qwen_pairwise_reward import (
    ScalarRewardHead,
    bradley_terry_loss,
)


def test_bradley_terry_prefers_positive_margin():
    good_loss = bradley_terry_loss(
        torch.tensor([2.0]),
        torch.tensor([-1.0]),
    )
    bad_loss = bradley_terry_loss(
        torch.tensor([-1.0]),
        torch.tensor([2.0]),
    )

    assert good_loss.item() < bad_loss.item()


def test_zero_margin_loss_is_log_two():
    loss = bradley_terry_loss(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )

    assert torch.allclose(
        loss,
        torch.tensor(
            torch.log(torch.tensor(2.0))
        ),
    )


def test_scalar_reward_head_shape_and_grad():
    head = ScalarRewardHead(
        hidden_size=16,
        intermediate_size=8,
    )
    inputs = torch.randn(
        4,
        16,
        requires_grad=True,
    )
    scores = head(inputs)

    assert scores.shape == (4,)

    scores.sum().backward()
    assert inputs.grad is not None
