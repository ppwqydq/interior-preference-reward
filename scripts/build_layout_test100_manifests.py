#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""为 layout_test100 构建 Pointwise 和 Pairwise Manifest。"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "layout_test100"
)

IMAGE_PATTERN = re.compile(
    r"^(?P<room_id>\d+)_(?P<role>R|GOOD|BAD)$",
    re.IGNORECASE,
)

SUPPORTED_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

REQUIRED_ROLES = {
    "R",
    "GOOD",
    "BAD",
}


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument(
        "--expected_pairs",
        type=int,
        default=100,
    )

    return parser.parse_args()


def parse_image_identity(
    image_path: Path,
) -> tuple[str, str]:
    """从图片名解析 room_id 和角色。"""

    match = IMAGE_PATTERN.fullmatch(
        image_path.stem
    )

    if match is None:
        raise ValueError(
            "图片名不符合 <数字>_R/GOOD/BAD 格式："
            f"{image_path.name}"
        )

    room_id = match.group("room_id")
    role = match.group("role").upper()

    return room_id, role


def discover_image_groups(
    dataset_dir: Path,
) -> dict[str, dict[str, Path]]:
    """扫描目录并按 room_id 聚合图片。"""

    groups: dict[str, dict[str, Path]] = (
        defaultdict(dict)
    )

    image_paths = sorted(
        path
        for path in dataset_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_SUFFIXES
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"目录中没有找到图片：{dataset_dir}"
        )

    for image_path in image_paths:
        room_id, role = parse_image_identity(
            image_path
        )

        if role in groups[room_id]:
            raise RuntimeError(
                "发现重复角色图片："
                f"room_id={room_id}, "
                f"role={role}, "
                f"existing={groups[room_id][role]}, "
                f"duplicate={image_path}"
            )

        groups[room_id][role] = image_path

    return dict(groups)


def numeric_room_sort_key(
    room_id: str,
) -> int:
    """按数字房间 ID 排序。"""

    return int(room_id)


def validate_image_groups(
    groups: dict[str, dict[str, Path]],
    expected_pairs: int,
) -> list[str]:
    """校验数量和每组三张图片是否完整。"""

    incomplete: dict[str, list[str]] = {}

    for room_id, roles in groups.items():
        missing = sorted(
            REQUIRED_ROLES - set(roles)
        )

        if missing:
            incomplete[room_id] = missing

    if incomplete:
        raise RuntimeError(
            "发现不完整的图片组：\n"
            + json.dumps(
                incomplete,
                ensure_ascii=False,
                indent=2,
            )
        )

    room_ids = sorted(
        groups,
        key=numeric_room_sort_key,
    )

    if len(room_ids) != expected_pairs:
        raise RuntimeError(
            "完整 Pair 数量不符合预期："
            f"expected={expected_pairs}, "
            f"actual={len(room_ids)}"
        )

    return room_ids


def project_relative_path(
    path: Path,
) -> str:
    """转换为项目根目录相对路径。"""

    resolved_path = path.resolve()

    try:
        relative_path = (
            resolved_path.relative_to(
                PROJECT_ROOT.resolve()
            )
        )
    except ValueError as error:
        raise RuntimeError(
            f"图片不在项目目录中：{path}"
        ) from error

    return relative_path.as_posix()


def build_pairwise_row(
    room_id: str,
    roles: dict[str, Path],
) -> dict[str, Any]:
    """构建一条 Pairwise 记录。"""

    return {
        "pair_id": room_id,
        "room_id": room_id,
        "reference_image_path": (
            project_relative_path(
                roles["R"]
            )
        ),
        "positive_image_path": (
            project_relative_path(
                roles["GOOD"]
            )
        ),
        "negative_image_path": (
            project_relative_path(
                roles["BAD"]
            )
        ),
        "positive_role": "GOOD",
        "negative_role": "BAD",
        "expected_order": (
            "positive_score > negative_score"
        ),
    }


def build_pointwise_rows(
    room_id: str,
    roles: dict[str, Path],
) -> list[dict[str, Any]]:
    """构建 GOOD 和 BAD 两条 Pointwise 记录。"""

    reference_path = project_relative_path(
        roles["R"]
    )

    return [
        {
            "sample_id": (
                f"layout_test100:{room_id}:GOOD"
            ),
            "room_id": room_id,
            "reference_image_path": (
                reference_path
            ),
            "candidate_image_path": (
                project_relative_path(
                    roles["GOOD"]
                )
            ),
            "candidate_role": "GOOD",
            "label": 1,
            "label_text": "thumbs_up",
        },
        {
            "sample_id": (
                f"layout_test100:{room_id}:BAD"
            ),
            "room_id": room_id,
            "reference_image_path": (
                reference_path
            ),
            "candidate_image_path": (
                project_relative_path(
                    roles["BAD"]
                )
            ),
            "candidate_role": "BAD",
            "label": 0,
            "label_text": "dislike",
        },
    ]


def build_manifest_rows(
    groups: dict[str, dict[str, Path]],
    room_ids: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """构建完整 Pairwise 和 Pointwise 数据。"""

    pairwise_rows: list[dict[str, Any]] = []
    pointwise_rows: list[dict[str, Any]] = []

    for room_id in room_ids:
        roles = groups[room_id]

        pairwise_rows.append(
            build_pairwise_row(
                room_id,
                roles,
            )
        )

        pointwise_rows.extend(
            build_pointwise_rows(
                room_id,
                roles,
            )
        )

    return pairwise_rows, pointwise_rows


def atomic_write_jsonl(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """原子写入 JSONL 文件。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(
            temporary_file.name
        )

        for row in rows:
            temporary_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_path.replace(output_path)


def atomic_write_json(
    value: dict[str, Any],
    output_path: Path,
) -> None:
    """原子写入 JSON 文件。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(
            temporary_file.name
        )

        json.dump(
            value,
            temporary_file,
            ensure_ascii=False,
            indent=2,
        )
        temporary_file.write("\n")

    temporary_path.replace(output_path)


def main() -> None:
    """执行 Manifest 构建。"""

    args = parse_args()

    dataset_dir = (
        args.dataset_dir.resolve()
    )

    if not dataset_dir.is_dir():
        raise NotADirectoryError(
            dataset_dir
        )

    groups = discover_image_groups(
        dataset_dir
    )
    room_ids = validate_image_groups(
        groups,
        args.expected_pairs,
    )

    pairwise_rows, pointwise_rows = (
        build_manifest_rows(
            groups,
            room_ids,
        )
    )

    pairwise_path = (
        dataset_dir
        / "pairwise_test.jsonl"
    )
    pointwise_path = (
        dataset_dir
        / "pointwise_test.jsonl"
    )
    summary_path = (
        dataset_dir
        / "test_manifest_summary.json"
    )

    atomic_write_jsonl(
        pairwise_rows,
        pairwise_path,
    )
    atomic_write_jsonl(
        pointwise_rows,
        pointwise_path,
    )
    atomic_write_json(
        {
            "dataset_dir": str(
                dataset_dir
            ),
            "num_pairs": len(
                pairwise_rows
            ),
            "num_pointwise_samples": len(
                pointwise_rows
            ),
            "room_ids": room_ids,
            "pairwise_manifest": str(
                pairwise_path
            ),
            "pointwise_manifest": str(
                pointwise_path
            ),
            "policy": {
                "R": "reference image",
                "GOOD": "positive label 1",
                "BAD": "negative label 0",
                "primary_metric": (
                    "score_GOOD > score_BAD"
                ),
            },
        },
        summary_path,
    )

    print(
        f"完整 Pair：{len(pairwise_rows)}"
    )
    print(
        "Pointwise 样本："
        f"{len(pointwise_rows)}"
    )
    print(
        f"Pairwise：{pairwise_path}"
    )
    print(
        f"Pointwise：{pointwise_path}"
    )
    print(
        f"Summary：{summary_path}"
    )


if __name__ == "__main__":
    main()
