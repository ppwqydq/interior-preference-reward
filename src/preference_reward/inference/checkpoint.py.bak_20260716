#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reward Model Checkpoint 元数据读取工具。

每个训练 Epoch 目录中的 checkpoint_config.json
应完整记录推理所需的信息：

- 基础模型路径；
- Processor 路径；
- LoRA Adapter 路径；
- Prompt；
- 图片像素上限；
- A/B token ID；
- 负样本训练权重。

正式评估优先读取该文件，而不是重新依赖训练 YAML。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class RewardCheckpointConfig:
    """一份可直接用于推理的 Reward Checkpoint 配置。"""

    checkpoint_dir: Path
    base_model_path: Path
    processor_path: Path
    adapter_path: Path
    max_pixels: int
    token_a: int
    token_b: int
    negative_weight: float
    system_prompt: str
    user_prompt: str
    epoch: int | None
    raw: Dict[str, Any]
    use_room_type: bool = False
    room_type_prefix: str = "房型"


def resolve_checkpoint_path(
    checkpoint_dir: Path,
    value: str | Path,
) -> Path:
    """解析 checkpoint_config.json 中保存的路径。

    当前训练保存的是绝对路径；同时兼容未来可能保存的
    checkpoint 相对路径。
    """

    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        checkpoint_dir / path
    ).resolve()


def require_non_empty_string(
    config: Dict[str, Any],
    key: str,
    config_path: Path,
) -> str:
    """读取必需的非空字符串字段。"""

    value = config.get(key)

    if value is None or not str(value).strip():
        raise KeyError(
            f"{config_path} 缺少有效字段：{key}"
        )

    return str(value).strip()


def load_reward_checkpoint_config(
    checkpoint_dir: Path,
    validate_paths: bool = True,
) -> RewardCheckpointConfig:
    """读取并校验 Reward Model Checkpoint 元数据。"""

    checkpoint_dir = (
        checkpoint_dir
        .expanduser()
        .resolve()
    )

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint 目录不存在：{checkpoint_dir}"
        )

    config_path = (
        checkpoint_dir
        / "checkpoint_config.json"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"缺少 Checkpoint 配置：{config_path}"
        )

    try:
        raw = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Checkpoint 配置不是有效 JSON："
            f"{config_path}"
        ) from exc

    if not isinstance(raw, dict):
        raise TypeError(
            f"Checkpoint 配置根节点必须是对象："
            f"{config_path}"
        )

    base_model_path = resolve_checkpoint_path(
        checkpoint_dir,
        require_non_empty_string(
            raw,
            "base_model_path",
            config_path,
        ),
    )

    processor_path = resolve_checkpoint_path(
        checkpoint_dir,
        require_non_empty_string(
            raw,
            "processor_path",
            config_path,
        ),
    )

    adapter_path = resolve_checkpoint_path(
        checkpoint_dir,
        require_non_empty_string(
            raw,
            "adapter_path",
            config_path,
        ),
    )

    max_pixels = int(
        raw["max_pixels"]
    )

    if max_pixels <= 0:
        raise ValueError(
            f"max_pixels 必须大于 0："
            f"{max_pixels}"
        )

    token_ids = raw.get("token_ids")

    if not isinstance(token_ids, dict):
        raise TypeError(
            f"{config_path} 中 token_ids 必须是对象"
        )

    token_a = int(token_ids["A"])
    token_b = int(token_ids["B"])

    if token_a == token_b:
        raise ValueError(
            "A/B token ID 不能相同"
        )

    negative_weight = float(
        raw["negative_weight"]
    )

    if negative_weight <= 0:
        raise ValueError(
            "negative_weight 必须大于 0"
        )

    prompt = raw.get("prompt")

    if not isinstance(prompt, dict):
        raise TypeError(
            f"{config_path} 中 prompt 必须是对象"
        )

    system_prompt = str(
        prompt.get("system", "")
    ).strip()
    user_prompt = str(
        prompt.get("user", "")
    ).strip()

    # 旧 Checkpoint 中没有房型设置时，
    # 按原始双图 Baseline 方式加载。
    use_room_type = bool(
        prompt.get(
            "use_room_type",
            False,
        )
    )
    room_type_prefix = str(
        prompt.get(
            "room_type_prefix",
            "房型",
        )
    ).strip() or "房型"

    if not system_prompt:
        raise ValueError(
            "Checkpoint 中 system prompt 为空"
        )

    if not user_prompt:
        raise ValueError(
            "Checkpoint 中 user prompt 为空"
        )

    epoch_value = raw.get("epoch")
    epoch = (
        int(epoch_value)
        if epoch_value is not None
        else None
    )

    if validate_paths:
        required_paths = {
            "基础模型": base_model_path,
            "Processor": processor_path,
            "Adapter": adapter_path,
            "Adapter 配置": (
                adapter_path
                / "adapter_config.json"
            ),
            "Adapter 权重": (
                adapter_path
                / "adapter_model.safetensors"
            ),
        }

        missing = [
            f"{name}：{path}"
            for name, path
            in required_paths.items()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Checkpoint 依赖文件不完整：\n"
                + "\n".join(
                    f"  - {item}"
                    for item in missing
                )
            )

    return RewardCheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        base_model_path=base_model_path,
        processor_path=processor_path,
        adapter_path=adapter_path,
        max_pixels=max_pixels,
        token_a=token_a,
        token_b=token_b,
        negative_weight=negative_weight,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        epoch=epoch,
        raw=raw,
        use_room_type=use_room_type,
        room_type_prefix=room_type_prefix,
    )
