#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

from preference_reward.data.manifest import (
    read_preference_manifest,
)
from preference_reward.models.prompting import (
    compose_user_prompt,
    normalize_room_type,
)


def test_normalize_known_room_type() -> None:
    assert (
        normalize_room_type(
            " Living Room "
        )
        == "Living Room"
    )


def test_normalize_unknown_room_types() -> None:
    for value in (
        None,
        "",
        " ",
        "Unknown",
        " unknown ",
        "NONE",
        "null",
        "N/A",
    ):
        assert normalize_room_type(value) == ""


def test_known_room_type_is_added() -> None:
    result = compose_user_prompt(
        base_user_prompt=(
            "第一张图是原始空房间图。"
        ),
        room_type="Bedroom",
        use_room_type=True,
    )

    assert result == (
        "房型：Bedroom\n\n"
        "第一张图是原始空房间图。"
    )


def test_unknown_room_type_is_not_added() -> None:
    result = compose_user_prompt(
        base_user_prompt=(
            "第一张图是原始空房间图。"
        ),
        room_type="Unknown",
        use_room_type=True,
    )

    assert result == (
        "第一张图是原始空房间图。"
    )
    assert "房型" not in result
    assert "Unknown" not in result


def test_room_type_disabled() -> None:
    result = compose_user_prompt(
        base_user_prompt=(
            "第一张图是原始空房间图。"
        ),
        room_type="Kitchen",
        use_room_type=False,
    )

    assert result == (
        "第一张图是原始空房间图。"
    )


def test_manifest_reads_room_type(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "samples.jsonl"

    manifest_path.write_text(
        json.dumps(
            {
                "empty_room_image": (
                    "empty.png"
                ),
                "generated_furniture_image": (
                    "generated.png"
                ),
                "label": 1,
                "room_type": "Bathroom",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    samples = read_preference_manifest(
        manifest_path=manifest_path,
        project_root=tmp_path,
        validate_image_paths=False,
    )

    assert len(samples) == 1
    assert samples[0].room_type == "Bathroom"


def test_old_manifest_defaults_to_unknown(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "old.jsonl"

    manifest_path.write_text(
        json.dumps(
            {
                "empty_room_image": (
                    "empty.png"
                ),
                "generated_furniture_image": (
                    "generated.png"
                ),
                "label": 0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    samples = read_preference_manifest(
        manifest_path=manifest_path,
        project_root=tmp_path,
        validate_image_paths=False,
    )

    assert len(samples) == 1
    assert samples[0].room_type == "Unknown"
