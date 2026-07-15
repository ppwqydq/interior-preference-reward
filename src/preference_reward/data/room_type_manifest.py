#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""为现有偏好数据 Manifest 严格回填房型。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
from urllib.parse import urlparse


LABEL_MAPPING = {
    "thumbs up": 1,
    "dislike": 0,
}


def configure_csv_field_limit() -> None:
    """允许读取包含大字段的原始 CSV。"""

    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_text(value: object) -> str:
    """将可选字段规范化为去除首尾空白的字符串。"""

    return str(value or "").strip()


def is_http_url(value: str) -> bool:
    """检查字符串是否为 HTTP/HTTPS URL。"""

    if not value:
        return False

    parsed = urlparse(value)

    return (
        parsed.scheme.lower() in {"http", "https"}
        and bool(parsed.netloc)
    )


def sha256_text(value: str) -> str:
    """计算字符串 SHA256。"""

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def make_csv_key(
    empty_room_url: str,
    generated_furniture_url: str,
    label: int,
) -> tuple[str, str, int]:
    """根据原始 CSV 字段构造关联键。"""

    return (
        sha256_text(empty_room_url),
        sha256_text(generated_furniture_url),
        int(label),
    )


def make_manifest_key(
    row: Mapping[str, Any],
) -> tuple[str, str, int]:
    """根据 Manifest 本地图片路径构造关联键。"""

    return (
        Path(
            str(row["empty_room_image"])
        ).stem,
        Path(
            str(
                row["generated_furniture_image"]
            )
        ).stem,
        int(row["label"]),
    )


def write_jsonl_atomic(
    rows: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """原子写入 JSONL。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)

        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    os.replace(
        temporary_path,
        output_path,
    )


def write_json_atomic(
    value: Mapping[str, Any],
    output_path: Path,
) -> None:
    """原子写入 JSON。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)

        json.dump(
            dict(value),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    os.replace(
        temporary_path,
        output_path,
    )


def build_room_type_index(
    raw_dir: Path,
    csv_pattern: str = "*.csv",
    csv_encoding: str = "utf-8-sig",
) -> tuple[
    Dict[tuple[str, str, int], set[str]],
    Dict[str, Any],
]:
    """扫描原始 CSV，建立图片对、标签到房型的索引。"""

    configure_csv_field_limit()

    csv_paths = sorted(
        path
        for path in raw_dir.rglob(csv_pattern)
        if path.is_file()
    )

    if not csv_paths:
        raise FileNotFoundError(
            f"没有找到原始 CSV：{raw_dir}"
        )

    room_types_by_key: Dict[
        tuple[str, str, int],
        set[str],
    ] = defaultdict(set)

    total_rows = 0
    selected_rows = 0
    valid_url_rows = 0
    non_empty_room_type_rows = 0
    file_reports = []

    for csv_path in csv_paths:
        file_total = 0
        file_selected = 0
        file_valid_urls = 0
        file_room_types = 0

        with csv_path.open(
            "r",
            encoding=csv_encoding,
            errors="replace",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            required_columns = {
                "反馈行为原始值",
                "空房间图",
                "生成家具图",
                "房型",
            }

            fieldnames = set(
                reader.fieldnames or []
            )
            missing_columns = (
                required_columns - fieldnames
            )

            if missing_columns:
                raise KeyError(
                    f"{csv_path} 缺少字段："
                    f"{sorted(missing_columns)}"
                )

            for row in reader:
                total_rows += 1
                file_total += 1

                behavior = normalize_text(
                    row.get("反馈行为原始值")
                ).lower()

                if behavior not in LABEL_MAPPING:
                    continue

                selected_rows += 1
                file_selected += 1

                empty_room_url = normalize_text(
                    row.get("空房间图")
                )
                generated_furniture_url = (
                    normalize_text(
                        row.get("生成家具图")
                    )
                )

                if (
                    not is_http_url(empty_room_url)
                    or not is_http_url(
                        generated_furniture_url
                    )
                ):
                    continue

                valid_url_rows += 1
                file_valid_urls += 1

                room_type = normalize_text(
                    row.get("房型")
                )

                if not room_type:
                    continue

                non_empty_room_type_rows += 1
                file_room_types += 1

                key = make_csv_key(
                    empty_room_url=empty_room_url,
                    generated_furniture_url=(
                        generated_furniture_url
                    ),
                    label=LABEL_MAPPING[behavior],
                )

                room_types_by_key[key].add(
                    room_type
                )

        file_reports.append(
            {
                "file": str(csv_path),
                "total_rows": file_total,
                "selected_label_rows": (
                    file_selected
                ),
                "valid_image_url_rows": (
                    file_valid_urls
                ),
                "non_empty_room_type_rows": (
                    file_room_types
                ),
            }
        )

    conflict_keys = {
        key: sorted(values)
        for key, values
        in room_types_by_key.items()
        if len(values) > 1
    }

    report = {
        "raw_dir": str(raw_dir),
        "num_csv_files": len(csv_paths),
        "total_csv_rows": total_rows,
        "selected_label_rows": selected_rows,
        "valid_image_url_rows": valid_url_rows,
        "non_empty_room_type_rows": (
            non_empty_room_type_rows
        ),
        "indexed_keys": len(
            room_types_by_key
        ),
        "conflicting_keys": len(
            conflict_keys
        ),
        "files": file_reports,
        "conflict_examples": [
            {
                "empty_room_hash": key[0],
                "generated_furniture_hash": key[1],
                "label": key[2],
                "room_types": room_types,
            }
            for key, room_types in list(
                sorted(conflict_keys.items())
            )[:50]
        ],
    }

    return room_types_by_key, report


def augment_manifest(
    input_path: Path,
    output_path: Path,
    room_types_by_key: Mapping[
        tuple[str, str, int],
        set[str],
    ],
) -> Dict[str, Any]:
    """严格生成带 room_type 的平行 Manifest。"""

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Manifest 不存在：{input_path}"
        )

    source_rows = []
    output_rows = []

    missing_samples = []
    conflicting_samples = []

    room_type_counts = Counter()
    room_type_label_counts = Counter()
    label_counts = Counter()

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            row = json.loads(text)

            if not isinstance(row, dict):
                raise TypeError(
                    f"{input_path}:{line_number} "
                    "不是 JSON 对象"
                )

            source_rows.append(row)

            key = make_manifest_key(row)
            room_types = room_types_by_key.get(
                key,
                set(),
            )

            if not room_types:
                missing_samples.append(
                    {
                        "line": line_number,
                        "key": list(key),
                    }
                )
                continue

            if len(room_types) != 1:
                conflicting_samples.append(
                    {
                        "line": line_number,
                        "key": list(key),
                        "room_types": sorted(
                            room_types
                        ),
                    }
                )
                continue

            room_type = next(
                iter(room_types)
            )

            output_row = dict(row)
            output_row["room_type"] = (
                room_type
            )
            output_rows.append(output_row)

            label = int(row["label"])

            room_type_counts[room_type] += 1
            label_counts[label] += 1
            room_type_label_counts[
                (room_type, label)
            ] += 1

    if missing_samples or conflicting_samples:
        raise RuntimeError(
            f"{input_path} 房型关联不完整："
            f"missing={len(missing_samples)}, "
            f"conflicts="
            f"{len(conflicting_samples)}"
        )

    if len(source_rows) != len(output_rows):
        raise AssertionError(
            f"{input_path} 样本数发生变化："
            f"{len(source_rows)} -> "
            f"{len(output_rows)}"
        )

    # 校验除了新增 room_type 外，原字段和值均未变化。
    for source_row, output_row in zip(
        source_rows,
        output_rows,
    ):
        restored_row = dict(output_row)
        restored_row.pop(
            "room_type",
            None,
        )

        if restored_row != source_row:
            raise AssertionError(
                "回填房型时修改了原始字段"
            )

    write_jsonl_atomic(
        output_rows,
        output_path,
    )

    distribution = []

    for room_type, total in (
        room_type_counts.most_common()
    ):
        likes = room_type_label_counts[
            (room_type, 1)
        ]
        dislikes = room_type_label_counts[
            (room_type, 0)
        ]

        distribution.append(
            {
                "room_type": room_type,
                "total": total,
                "like": likes,
                "dislike": dislikes,
                "like_rate": likes / total,
            }
        )

    return {
        "input_manifest": str(input_path),
        "output_manifest": str(output_path),
        "rows": len(output_rows),
        "label_counts": {
            "0": int(label_counts.get(0, 0)),
            "1": int(label_counts.get(1, 0)),
        },
        "missing_room_type": 0,
        "conflicting_room_type": 0,
        "room_type_distribution": distribution,
    }
