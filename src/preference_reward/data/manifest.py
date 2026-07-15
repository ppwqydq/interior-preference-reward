#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""双图偏好训练清单读取与校验。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class PreferenceSample:
    """一条双图偏好训练样本。"""

    sample_id: str
    empty_room_image: Path
    generated_furniture_image: Path
    label: int
    room_type: str = "Unknown"


def resolve_image_path(
    value: str,
    project_root: Path,
) -> Path:
    """解析清单中的图片路径。"""

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def read_preference_manifest(
    manifest_path: Path,
    project_root: Path,
    validate_image_paths: bool = True,
) -> List[PreferenceSample]:
    """读取训练或验证 JSONL。"""

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"数据清单不存在：{manifest_path}"
        )

    samples: List[PreferenceSample] = []
    missing_paths: List[str] = []

    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()

            if not text:
                continue

            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON 解析失败："
                    f"{manifest_path}:{line_number}"
                ) from exc

            required = {
                "empty_room_image",
                "generated_furniture_image",
                "label",
            }
            missing_fields = required - set(row)

            if missing_fields:
                raise KeyError(
                    f"{manifest_path}:{line_number} "
                    f"缺少字段：{sorted(missing_fields)}"
                )

            label = int(row["label"])

            if label not in (0, 1):
                raise ValueError(
                    f"{manifest_path}:{line_number} "
                    f"标签必须为 0 或 1，实际为 {label}"
                )

            empty_room_image = resolve_image_path(
                str(row["empty_room_image"]),
                project_root,
            )
            generated_furniture_image = resolve_image_path(
                str(row["generated_furniture_image"]),
                project_root,
            )

            if validate_image_paths:
                if not empty_room_image.is_file():
                    missing_paths.append(str(empty_room_image))

                if not generated_furniture_image.is_file():
                    missing_paths.append(
                        str(generated_furniture_image)
                    )

            sample_id = str(
                row.get("sample_id")
                or f"{manifest_path.stem}:{line_number}"
            )

            room_type = str(
                row.get("room_type") or ""
            ).strip() or "Unknown"

            samples.append(
                PreferenceSample(
                    sample_id=sample_id,
                    empty_room_image=empty_room_image,
                    generated_furniture_image=(
                        generated_furniture_image
                    ),
                    label=label,
                    room_type=room_type,
                )
            )

    if not samples:
        raise RuntimeError(
            f"数据清单中没有有效样本：{manifest_path}"
        )

    if missing_paths:
        preview = "\n".join(
            f"  - {path}"
            for path in missing_paths[:20]
        )
        raise FileNotFoundError(
            f"共有 {len(missing_paths)} 个图片路径不存在：\n"
            f"{preview}"
        )

    return samples


def batched(
    samples: Sequence[PreferenceSample],
    batch_size: int,
) -> Iterable[Sequence[PreferenceSample]]:
    """按指定大小切分 batch。"""

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    for start in range(0, len(samples), batch_size):
        yield samples[start:start + batch_size]


def count_labels(
    samples: Sequence[PreferenceSample],
) -> Dict[str, int]:
    """统计正负样本数量。"""

    counts = Counter(sample.label for sample in samples)

    return {
        "0": int(counts.get(0, 0)),
        "1": int(counts.get(1, 0)),
    }
