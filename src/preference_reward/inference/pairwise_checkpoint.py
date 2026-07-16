#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""加载新格式 Qwen Pairwise Reward Checkpoint。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor

from preference_reward.models.qwen_ab_reward import (
    QwenABRewardBackend,
    get_single_token_id,
)
from preference_reward.models.qwen_pairwise_reward import (
    SCORE_TYPE_SCALAR_HEAD,
    ScalarRewardHead,
)
from preference_reward.inference.qwen_ab_loader import (
    load_base_model,
)


@dataclass(frozen=True)
class PairwiseCheckpoint:
    checkpoint_dir: Path
    base_model_path: Path
    processor_path: Path
    adapter_path: Path
    reward_head_path: Path | None
    score_type: str
    max_pixels: int
    token_a: int
    token_b: int
    system_prompt: str
    user_prompt: str
    scalar_head: dict[str, Any]
    epoch: int | None
    raw: dict[str, Any]


def _resolve(
    checkpoint_dir: Path,
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = checkpoint_dir / path

    return path.resolve()


def load_pairwise_checkpoint_config(
    checkpoint_dir: Path,
    validate_paths: bool = True,
) -> PairwiseCheckpoint:
    checkpoint_dir = (
        checkpoint_dir.expanduser().resolve()
    )
    config_path = (
        checkpoint_dir
        / "checkpoint_config.json"
    )

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    raw = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        raw.get("checkpoint_format")
        != "qwen_pairwise_reward_v1"
    ):
        raise ValueError(
            "不是 qwen_pairwise_reward_v1 "
            f"Checkpoint：{config_path}"
        )

    prompt = dict(raw["prompt"])
    token_ids = dict(raw["token_ids"])
    head_value = raw.get("reward_head_path")

    checkpoint = PairwiseCheckpoint(
        checkpoint_dir=checkpoint_dir,
        base_model_path=_resolve(
            checkpoint_dir,
            raw["base_model_path"],
        ),
        processor_path=_resolve(
            checkpoint_dir,
            raw["processor_path"],
        ),
        adapter_path=_resolve(
            checkpoint_dir,
            raw["adapter_path"],
        ),
        reward_head_path=(
            _resolve(
                checkpoint_dir,
                head_value,
            )
            if head_value
            else None
        ),
        score_type=str(raw["score_type"]),
        max_pixels=int(raw["max_pixels"]),
        token_a=int(token_ids["A"]),
        token_b=int(token_ids["B"]),
        system_prompt=str(prompt["system"]),
        user_prompt=str(prompt["user"]),
        scalar_head=dict(
            raw.get("scalar_head") or {}
        ),
        epoch=(
            int(raw["epoch"])
            if raw.get("epoch") is not None
            else None
        ),
        raw=raw,
    )

    if validate_paths:
        required = [
            checkpoint.base_model_path,
            checkpoint.processor_path,
            checkpoint.adapter_path
            / "adapter_config.json",
            checkpoint.adapter_path
            / "adapter_model.safetensors",
        ]

        if (
            checkpoint.score_type
            == SCORE_TYPE_SCALAR_HEAD
        ):
            if (
                checkpoint.reward_head_path
                is None
            ):
                raise FileNotFoundError(
                    "Scalar Checkpoint 缺少 "
                    "reward_head_path"
                )
            required.append(
                checkpoint.reward_head_path
            )

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Checkpoint 文件不完整：\n"
                + "\n".join(missing)
            )

    return checkpoint


def load_pairwise_backend(
    checkpoint: PairwiseCheckpoint,
    device: torch.device,
    logger: Any,
    attn_implementation: str = "",
) -> tuple[
    QwenABRewardBackend,
    ScalarRewardHead | None,
]:
    processor = AutoProcessor.from_pretrained(
        str(checkpoint.processor_path),
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "right"

    if (
        processor.tokenizer.pad_token_id is None
        and processor.tokenizer.eos_token
        is not None
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

    if token_a != checkpoint.token_a:
        raise RuntimeError(
            "A token ID 与 Checkpoint 不一致"
        )
    if token_b != checkpoint.token_b:
        raise RuntimeError(
            "B token ID 与 Checkpoint 不一致"
        )

    base_model = load_base_model(
        model_path=checkpoint.base_model_path,
        device=device,
        attn_implementation=(
            attn_implementation
        ),
        logger=logger,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint.adapter_path),
        is_trainable=False,
    )
    model.to(device)
    model.eval()

    backend = QwenABRewardBackend(
        model=model,
        processor=processor,
        device=device,
        system_prompt=checkpoint.system_prompt,
        user_prompt=checkpoint.user_prompt,
        max_pixels=checkpoint.max_pixels,
        token_a=token_a,
        token_b=token_b,
    )

    reward_head: ScalarRewardHead | None = None

    if (
        checkpoint.score_type
        == SCORE_TYPE_SCALAR_HEAD
    ):
        reward_head = ScalarRewardHead(
            hidden_size=int(
                checkpoint.scalar_head[
                    "hidden_size"
                ]
            ),
            intermediate_size=int(
                checkpoint.scalar_head[
                    "intermediate_size"
                ]
            ),
        ).to(
            device=device,
            dtype=torch.float32,
        )
        state = torch.load(
            checkpoint.reward_head_path,
            map_location=device,
            weights_only=True,
        )
        reward_head.load_state_dict(state)
        reward_head.eval()

    return backend, reward_head
