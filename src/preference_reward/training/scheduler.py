#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""学习率控制与 Early Stopping。

训练策略：

1. 前若干 optimizer step 线性 Warmup；
2. Warmup 后保持当前学习率；
3. 验证指标连续若干 epoch 未改善时降低学习率；
4. 更长时间未改善时提前停止训练。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


def get_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> float:
    """读取第一个参数组的当前学习率。"""

    if not optimizer.param_groups:
        raise RuntimeError("Optimizer 中没有参数组")

    return float(
        optimizer.param_groups[0]["lr"]
    )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    """设置所有参数组的学习率。"""

    if learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = float(
            learning_rate
        )


@dataclass
class LinearWarmupController:
    """固定步数的线性 Warmup。

    optimizer step 从 1 开始计数：

        step 1             -> base_lr / warmup_steps
        step warmup_steps  -> base_lr
        后续 step          -> 不再主动修改学习率

    Warmup 完成后，学习率由 Plateau 控制器管理。
    """

    base_learning_rate: float
    warmup_steps: int

    def __post_init__(self) -> None:
        if self.base_learning_rate <= 0:
            raise ValueError(
                "base_learning_rate 必须大于 0"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps 不能小于 0"
            )

    def learning_rate_for_step(
        self,
        optimizer_step: int,
    ) -> float:
        """返回指定 optimizer step 应使用的学习率。"""

        if self.warmup_steps == 0:
            return self.base_learning_rate

        bounded_step = min(
            max(int(optimizer_step), 1),
            self.warmup_steps,
        )

        return (
            self.base_learning_rate
            * bounded_step
            / self.warmup_steps
        )

    def initialize(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """设置第一个 optimizer step 的学习率。"""

        initial_learning_rate = (
            self.learning_rate_for_step(1)
        )

        set_optimizer_learning_rate(
            optimizer,
            initial_learning_rate,
        )

        return initial_learning_rate

    def advance_after_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        completed_optimizer_steps: int,
    ) -> float:
        """完成一次参数更新后，设置下一次更新的学习率。

        Warmup 已完成时不再覆盖 Plateau 设置的学习率。
        """

        if (
            self.warmup_steps == 0
            or completed_optimizer_steps
            >= self.warmup_steps
        ):
            return get_optimizer_learning_rate(
                optimizer
            )

        next_step = completed_optimizer_steps + 1
        next_learning_rate = (
            self.learning_rate_for_step(
                next_step
            )
        )

        set_optimizer_learning_rate(
            optimizer,
            next_learning_rate,
        )

        return next_learning_rate

    def is_complete(
        self,
        completed_optimizer_steps: int,
    ) -> bool:
        """判断 Warmup 是否已经完成。"""

        return (
            self.warmup_steps == 0
            or completed_optimizer_steps
            >= self.warmup_steps
        )


@dataclass
class PlateauLrController:
    """验证指标停滞时降低学习率。

    这里自行实现 patience 逻辑，语义明确为：

        连续 patience 个 epoch 未有效改善
        -> 立即降低一次学习率
    """

    mode: str
    factor: float
    patience: int
    threshold: float
    minimum_learning_rate: float

    best_value: float | None = None
    bad_epochs: int = 0
    reduction_count: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError(
                "mode 必须是 max 或 min"
            )

        if not 0.0 < self.factor < 1.0:
            raise ValueError(
                "factor 必须位于 (0, 1)"
            )

        if self.patience <= 0:
            raise ValueError(
                "patience 必须大于 0"
            )

        if self.threshold < 0:
            raise ValueError(
                "threshold 不能小于 0"
            )

        if self.minimum_learning_rate <= 0:
            raise ValueError(
                "minimum_learning_rate 必须大于 0"
            )

    def is_improvement(
        self,
        value: float,
    ) -> bool:
        """判断指标是否出现有效改善。"""

        if self.best_value is None:
            return True

        if self.mode == "max":
            return (
                value
                > self.best_value
                + self.threshold
            )

        return (
            value
            < self.best_value
            - self.threshold
        )

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        value: float,
    ) -> Dict[str, Any]:
        """根据本轮验证指标更新学习率。"""

        old_learning_rate = (
            get_optimizer_learning_rate(
                optimizer
            )
        )

        improved = self.is_improvement(value)
        reduced = False

        if improved:
            self.best_value = float(value)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            new_learning_rate = max(
                self.minimum_learning_rate,
                old_learning_rate
                * self.factor,
            )

            if (
                new_learning_rate
                < old_learning_rate
                - 1.0e-15
            ):
                set_optimizer_learning_rate(
                    optimizer,
                    new_learning_rate,
                )
                reduced = True
                self.reduction_count += 1

            # 无论是否已经达到最小学习率，
            # 都重新开始计算下一段停滞周期。
            self.bad_epochs = 0

        return {
            "improved": improved,
            "best_value": self.best_value,
            "bad_epochs": self.bad_epochs,
            "reduced": reduced,
            "reduction_count": (
                self.reduction_count
            ),
            "old_learning_rate": (
                old_learning_rate
            ),
            "new_learning_rate": (
                get_optimizer_learning_rate(
                    optimizer
                )
            ),
        }


@dataclass
class EarlyStoppingController:
    """基于验证指标的 Early Stopping。"""

    enabled: bool
    mode: str
    minimum_epochs: int
    patience: int
    min_delta: float

    best_value: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError(
                "mode 必须是 max 或 min"
            )

        if self.minimum_epochs < 1:
            raise ValueError(
                "minimum_epochs 必须大于等于 1"
            )

        if self.patience <= 0:
            raise ValueError(
                "patience 必须大于 0"
            )

        if self.min_delta < 0:
            raise ValueError(
                "min_delta 不能小于 0"
            )

    def is_improvement(
        self,
        value: float,
    ) -> bool:
        """判断是否出现足够大的指标改善。"""

        if self.best_value is None:
            return True

        if self.mode == "max":
            return (
                value
                > self.best_value
                + self.min_delta
            )

        return (
            value
            < self.best_value
            - self.min_delta
        )

    def step(
        self,
        epoch: int,
        value: float,
    ) -> Dict[str, Any]:
        """更新 Early Stopping 状态。"""

        improved = self.is_improvement(value)

        if improved:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = (
            self.enabled
            and epoch >= self.minimum_epochs
            and self.bad_epochs
            >= self.patience
        )

        return {
            "improved": improved,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "should_stop": should_stop,
        }
