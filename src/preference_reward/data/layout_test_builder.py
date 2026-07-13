#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""构建空间布局外部测试集。

目录中的图片命名规则：

    <room_id>_R.png
    <room_id>_GOOD.png
    <room_id>_BAD.png

输出两份清单：

1. Pointwise：
   R + GOOD，label=1
   R + BAD，label=0

2. Pairwise：
   同一个房间下比较 GOOD 和 BAD 的模型得分。

同时使用文件内容 SHA256 检查测试图片是否与训练数据重复，
避免训练集和外部测试集发生数据泄漏。
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

REQUIRED_ROLES = {
    "R",
    "GOOD",
    "BAD",
}


def room_sort_key(room_id: str) -> tuple[int, int | str]:
    """数字房间 ID 按数值排序，其他 ID 按字符串排序。"""

    if room_id.isdigit():
        return 0, int(room_id)

    return 1, room_id


def project_relative_path(
    path: Path,
    project_root: Path,
) -> str:
    """返回相对于项目根目录的 POSIX 路径。"""

    return (
        path.resolve()
        .relative_to(project_root.resolve())
        .as_posix()
    )


def file_sha256(path: Path) -> str:
    """分块计算文件内容 SHA256。"""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def scan_layout_groups(
    source_dir: Path,
) -> tuple[Dict[str, Dict[str, Path]], List[str]]:
    """扫描测试目录并识别 R、GOOD、BAD 图片。

    返回：
    - 房间图片分组；
    - 未参与测试集构建的文件名。
    """

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"测试集目录不存在：{source_dir}"
        )

    groups: Dict[str, Dict[str, Path]] = defaultdict(dict)
    ignored_files: List[str] = []

    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            ignored_files.append(path.name)
            continue

        matched = False

        for suffix, role in (
            ("_GOOD", "GOOD"),
            ("_BAD", "BAD"),
            ("_R", "R"),
        ):
            if not path.stem.endswith(suffix):
                continue

            room_id = path.stem[:-len(suffix)]

            if not room_id:
                raise ValueError(
                    f"无法识别房间 ID：{path.name}"
                )

            if role in groups[room_id]:
                raise RuntimeError(
                    f"同一房间存在多个 {role} 图片："
                    f"{groups[room_id][role]} 与 {path}"
                )

            groups[room_id][role] = path
            matched = True
            break

        if not matched:
            ignored_files.append(path.name)

    if not groups:
        raise RuntimeError(
            f"没有识别到布局测试图片：{source_dir}"
        )

    incomplete_groups = {
        room_id: sorted(
            REQUIRED_ROLES - set(role_paths)
        )
        for room_id, role_paths in groups.items()
        if set(role_paths) != REQUIRED_ROLES
    }

    if incomplete_groups:
        raise RuntimeError(
            "存在不完整的 R/GOOD/BAD 分组：\n"
            + json.dumps(
                incomplete_groups,
                ensure_ascii=False,
                indent=2,
            )
        )

    return dict(groups), ignored_files


def build_manifest_rows(
    groups: Mapping[str, Mapping[str, Path]],
    project_root: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """生成 pointwise 和 pairwise 清单。"""

    pointwise_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []

    for room_id in sorted(groups, key=room_sort_key):
        role_paths = groups[room_id]

        reference_path = project_relative_path(
            role_paths["R"],
            project_root,
        )
        good_path = project_relative_path(
            role_paths["GOOD"],
            project_root,
        )
        bad_path = project_relative_path(
            role_paths["BAD"],
            project_root,
        )

        pointwise_rows.extend(
            [
                {
                    "sample_id": f"{room_id}_GOOD",
                    "room_id": room_id,
                    "candidate_role": "GOOD",
                    "reference_image_path": reference_path,
                    "candidate_image_path": good_path,
                    "label": 1,
                },
                {
                    "sample_id": f"{room_id}_BAD",
                    "room_id": room_id,
                    "candidate_role": "BAD",
                    "reference_image_path": reference_path,
                    "candidate_image_path": bad_path,
                    "label": 0,
                },
            ]
        )

        pairwise_rows.append(
            {
                "pair_id": room_id,
                "room_id": room_id,
                "reference_image_path": reference_path,
                "positive_image_path": good_path,
                "negative_image_path": bad_path,
            }
        )

    return pointwise_rows, pairwise_rows


def read_training_image_paths(
    manifest_path: Path,
    project_root: Path,
) -> List[Path]:
    """读取训练数据清单中的全部图片路径。"""

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"训练数据清单不存在：{manifest_path}"
        )

    image_paths: set[Path] = set()

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
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

            for field in (
                "empty_room_image",
                "generated_furniture_image",
            ):
                value = str(row.get(field, "")).strip()

                if not value:
                    raise KeyError(
                        f"{manifest_path}:{line_number} "
                        f"缺少字段 {field}"
                    )

                path = Path(value)

                if not path.is_absolute():
                    path = project_root / path

                path = path.resolve()

                if not path.is_file():
                    raise FileNotFoundError(
                        f"训练图片不存在：{path}"
                    )

                image_paths.add(path)

    return sorted(image_paths)


def build_hash_index(
    paths: Iterable[Path],
) -> Dict[str, List[str]]:
    """按照图片内容 SHA256 建立路径索引。"""

    index: Dict[str, List[str]] = defaultdict(list)

    for path in paths:
        index[file_sha256(path)].append(str(path))

    return dict(index)


def inspect_content_overlap(
    groups: Mapping[str, Mapping[str, Path]],
    training_manifest: Path,
    project_root: Path,
) -> Dict[str, Any]:
    """检查测试图片和训练图片是否内容重复。"""

    training_paths = read_training_image_paths(
        training_manifest,
        project_root,
    )
    training_hash_index = build_hash_index(training_paths)

    test_records: List[Dict[str, Any]] = []
    test_hash_roles: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for room_id in sorted(groups, key=room_sort_key):
        for role in ("R", "GOOD", "BAD"):
            path = groups[room_id][role]
            digest = file_sha256(path)

            record = {
                "room_id": room_id,
                "role": role,
                "path": str(path.resolve()),
                "sha256": digest,
            }

            test_records.append(record)
            test_hash_roles[digest].append(
                {
                    "room_id": room_id,
                    "role": role,
                    "path": str(path.resolve()),
                }
            )

    overlaps: List[Dict[str, Any]] = []

    for record in test_records:
        training_matches = training_hash_index.get(
            record["sha256"],
            [],
        )

        if not training_matches:
            continue

        overlaps.append(
            {
                **record,
                "matching_training_paths": training_matches,
            }
        )

    duplicate_test_contents = [
        {
            "sha256": digest,
            "test_images": records,
        }
        for digest, records in test_hash_roles.items()
        if len(records) > 1
    ]

    return {
        "training_unique_images": len(training_paths),
        "test_images": len(test_records),
        "train_test_overlap_count": len(overlaps),
        "train_test_overlaps": overlaps,
        "duplicate_content_groups_inside_test": len(
            duplicate_test_contents
        ),
        "duplicate_test_contents": duplicate_test_contents,
    }


def build_layout_test(
    project_root: Path,
    source_dir: Path,
    training_manifest: Path,
    pointwise_output: Path,
    pairwise_output: Path,
    report_output: Path,
    allow_training_overlap: bool = False,
) -> Dict[str, Any]:
    """执行完整的布局测试集构建。"""

    groups, ignored_files = scan_layout_groups(
        source_dir
    )

    pointwise_rows, pairwise_rows = build_manifest_rows(
        groups=groups,
        project_root=project_root,
    )

    overlap_report = inspect_content_overlap(
        groups=groups,
        training_manifest=training_manifest,
        project_root=project_root,
    )

    report = {
        "source_dir": str(source_dir.resolve()),
        "num_complete_groups": len(groups),
        "num_pointwise_samples": len(pointwise_rows),
        "num_pairwise_samples": len(pairwise_rows),
        "pointwise_label_counts": {
            "0": sum(
                row["label"] == 0
                for row in pointwise_rows
            ),
            "1": sum(
                row["label"] == 1
                for row in pointwise_rows
            ),
        },
        "ignored_files": ignored_files,
        "training_manifest": str(
            training_manifest.resolve()
        ),
        "overlap_check": overlap_report,
        "pointwise_output": str(
            pointwise_output.resolve()
        ),
        "pairwise_output": str(
            pairwise_output.resolve()
        ),
        "test_policy": {
            "primary_metric": (
                "pairwise_accuracy: "
                "score(GOOD) > score(BAD)"
            ),
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "used_for_threshold_selection": False,
        },
    }

    # 即使发现重叠，也先保存报告，方便定位问题。
    write_json_atomic(
        report,
        report_output,
    )

    overlap_count = overlap_report[
        "train_test_overlap_count"
    ]

    if overlap_count > 0 and not allow_training_overlap:
        raise RuntimeError(
            f"发现 {overlap_count} 张测试图片与训练数据内容重复。"
            f"详情见：{report_output}"
        )

    write_jsonl_atomic(
        pointwise_rows,
        pointwise_output,
    )
    write_jsonl_atomic(
        pairwise_rows,
        pairwise_output,
    )

    return report
