#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

from preference_reward.inference.checkpoint import (
    load_reward_checkpoint_config,
)


def write_checkpoint_config(
    checkpoint_dir: Path,
    prompt: dict,
) -> None:
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "base_model_path": "base_model",
        "processor_path": "processor",
        "adapter_path": "adapter",
        "max_pixels": 262144,
        "token_ids": {
            "A": 1,
            "B": 2,
        },
        "negative_weight": 2.0,
        "prompt": prompt,
        "epoch": 1,
    }

    (
        checkpoint_dir
        / "checkpoint_config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_old_checkpoint_disables_room_type(
    tmp_path: Path,
) -> None:
    checkpoint_dir = (
        tmp_path / "old_checkpoint"
    )

    write_checkpoint_config(
        checkpoint_dir=checkpoint_dir,
        prompt={
            "system": "system prompt",
            "user": "user prompt",
        },
    )

    checkpoint = (
        load_reward_checkpoint_config(
            checkpoint_dir=checkpoint_dir,
            validate_paths=False,
        )
    )

    assert checkpoint.use_room_type is False
    assert checkpoint.room_type_prefix == "房型"


def test_room_type_checkpoint_is_restored(
    tmp_path: Path,
) -> None:
    checkpoint_dir = (
        tmp_path / "room_type_checkpoint"
    )

    write_checkpoint_config(
        checkpoint_dir=checkpoint_dir,
        prompt={
            "system": "system prompt",
            "user": "user prompt",
            "use_room_type": True,
            "room_type_prefix": "房型",
        },
    )

    checkpoint = (
        load_reward_checkpoint_config(
            checkpoint_dir=checkpoint_dir,
            validate_paths=False,
        )
    )

    assert checkpoint.use_room_type is True
    assert checkpoint.room_type_prefix == "房型"


def test_empty_prefix_falls_back_to_default(
    tmp_path: Path,
) -> None:
    checkpoint_dir = (
        tmp_path / "empty_prefix_checkpoint"
    )

    write_checkpoint_config(
        checkpoint_dir=checkpoint_dir,
        prompt={
            "system": "system prompt",
            "user": "user prompt",
            "use_room_type": True,
            "room_type_prefix": "",
        },
    )

    checkpoint = (
        load_reward_checkpoint_config(
            checkpoint_dir=checkpoint_dir,
            validate_paths=False,
        )
    )

    assert checkpoint.use_room_type is True
    assert checkpoint.room_type_prefix == "房型"
