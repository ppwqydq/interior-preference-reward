#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""为当前 936 条偏好数据生成带房型的平行 Manifest。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )

from preference_reward.data.room_type_manifest import (
    augment_manifest,
    build_room_type_index,
    write_json_atomic,
)


MANIFEST_PAIRS = (
    (
        PROJECT_ROOT
        / "data/processed/samples.jsonl",
        PROJECT_ROOT
        / "data/processed/samples_room_type.jsonl",
    ),
    (
        PROJECT_ROOT
        / "data/splits/train.jsonl",
        PROJECT_ROOT
        / "data/splits/train_room_type.jsonl",
    ),
    (
        PROJECT_ROOT
        / "data/splits/val.jsonl",
        PROJECT_ROOT
        / "data/splits/val_room_type.jsonl",
    ),
)


def main() -> None:
    raw_dir = PROJECT_ROOT / "data/raw"

    room_types_by_key, raw_report = (
        build_room_type_index(
            raw_dir=raw_dir,
        )
    )

    manifest_reports = []

    for input_path, output_path in (
        MANIFEST_PAIRS
    ):
        manifest_reports.append(
            augment_manifest(
                input_path=input_path,
                output_path=output_path,
                room_types_by_key=(
                    room_types_by_key
                ),
            )
        )

    report = {
        "matching_key": (
            "SHA256(empty_room_url), "
            "SHA256(generated_furniture_url), "
            "label"
        ),
        "strict_mode": True,
        "original_manifests_overwritten": (
            False
        ),
        "raw_scan": raw_report,
        "manifests": manifest_reports,
    }

    report_path = (
        PROJECT_ROOT
        / "data/reports/"
        "room_type_manifest_report.json"
    )

    write_json_atomic(
        report,
        report_path,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
