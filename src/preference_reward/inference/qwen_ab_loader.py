#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Qwen3-VL A/B Reward Model 推理加载模块。

该模块只负责：

1. 从 Checkpoint 配置恢复 Processor；
2. 加载 Qwen3-VL 基础模型；
3. 加载已经训练完成的 LoRA Adapter；
4. 校验 A/B token ID；
5. 创建可供推理和评估使用的 QwenABRewardBackend。

本模块不包含训练逻辑，也不修改训练 Backend。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor

from preference_reward.inference.checkpoint import (
    RewardCheckpointConfig,
)
from preference_reward.models.qwen_ab_reward import (
    QwenABRewardBackend,
    get_model_class,
    get_single_token_id,
)


def validate_inference_paths(
    checkpoint: RewardCheckpointConfig,
) -> None:
    """检查推理依赖文件是否完整。"""

    required_paths = {
        "基础模型": checkpoint.base_model_path,
        "Processor": checkpoint.processor_path,
        "Adapter": checkpoint.adapter_path,
        "Adapter 配置": (
            checkpoint.adapter_path
            / "adapter_config.json"
        ),
        "Adapter 权重": (
            checkpoint.adapter_path
            / "adapter_model.safetensors"
        ),
    }

    missing_paths = [
        f"{name}：{path}"
        for name, path in required_paths.items()
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "推理依赖文件不完整：\n"
            + "\n".join(
                f"  - {item}"
                for item in missing_paths
            )
        )


def load_processor(
    checkpoint: RewardCheckpointConfig,
    logger: Any,
) -> tuple[Any, int, int]:
    """加载保存的 Processor，并校验 A/B token ID。"""

    logger.info(
        "加载推理 Processor：%s",
        checkpoint.processor_path,
    )

    processor = AutoProcessor.from_pretrained(
        str(checkpoint.processor_path),
        trust_remote_code=True,
    )

    processor.tokenizer.padding_side = "right"

    if (
        processor.tokenizer.pad_token_id is None
        and processor.tokenizer.eos_token is not None
    ):
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    token_a = get_single_token_id(
        processor.tokenizer,
        "A",
    )
    token_b = get_single_token_id(
        processor.tokenizer,
        "B",
    )

    logger.info(
        "推理 A/B token ID：A=%d，B=%d",
        token_a,
        token_b,
    )

    if token_a != checkpoint.token_a:
        raise RuntimeError(
            "A token ID 与训练 Checkpoint 不一致："
            f"当前={token_a}，"
            f"训练时={checkpoint.token_a}"
        )

    if token_b != checkpoint.token_b:
        raise RuntimeError(
            "B token ID 与训练 Checkpoint 不一致："
            f"当前={token_b}，"
            f"训练时={checkpoint.token_b}"
        )

    return processor, token_a, token_b


def load_base_model(
    model_path: Path,
    device: torch.device,
    attn_implementation: str,
    logger: Any,
) -> torch.nn.Module:
    """加载 BF16 Qwen3-VL 基础模型。"""

    model_class = get_model_class()

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if attn_implementation:
        load_kwargs["attn_implementation"] = (
            attn_implementation
        )

    logger.info(
        "加载推理基础模型：%s",
        model_path,
    )

    try:
        base_model = model_class.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            **load_kwargs,
        )
    except TypeError:
        # 兼容较旧版本 Transformers。
        base_model = model_class.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            **load_kwargs,
        )

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    base_model.to(device)

    return base_model


def load_qwen_ab_backend(
    checkpoint: RewardCheckpointConfig,
    device: torch.device,
    logger: Any,
    attn_implementation: str = "",
) -> QwenABRewardBackend:
    """加载可用于评估的完整 Qwen A/B Reward Backend。"""

    validate_inference_paths(checkpoint)

    processor, token_a, token_b = load_processor(
        checkpoint=checkpoint,
        logger=logger,
    )

    base_model = load_base_model(
        model_path=checkpoint.base_model_path,
        device=device,
        attn_implementation=attn_implementation,
        logger=logger,
    )

    logger.info(
        "加载 LoRA Adapter：%s",
        checkpoint.adapter_path,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint.adapter_path),
        is_trainable=False,
    )

    model.to(device)
    model.eval()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    logger.info(
        "推理模型加载完成："
        "trainable=%d，total=%d",
        trainable_parameters,
        total_parameters,
    )

    if trainable_parameters != 0:
        raise RuntimeError(
            "推理模型仍然存在可训练参数："
            f"{trainable_parameters}"
        )

    return QwenABRewardBackend(
        model=model,
        processor=processor,
        device=device,
        system_prompt=checkpoint.system_prompt,
        user_prompt=checkpoint.user_prompt,
        max_pixels=checkpoint.max_pixels,
        token_a=token_a,
        token_b=token_b,
    )
