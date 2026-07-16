#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""将 Layout100 按 Pair 划分为专项训练和验证 Manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "layout_test100"
    / "pairwise_test.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "layout100_curated"
)

PAIRWISE_REQUIRED_FIELDS = {
    "pair_id",
    "room_id",
    "reference_image_path",
    "positive_image_path",
    "negative_image_path",
}

IMAGE_FIELDS = (
    "reference_image_path",
    "positive_image_path",
    "negative_image_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按图片内容隔离 Layout100 Pair，构建 80/20 "
            "Pointwise 和 Pairwise Manifest。"
        )
    )
    parser.add_argument(
        "--source_manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--expected_pairs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--val_pairs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def resolve_image_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            text = line.strip()
            if not text:
                continue

            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON 解析失败：{path}:{line_number}"
                ) from exc

            missing = PAIRWISE_REQUIRED_FIELDS - set(row)
            if missing:
                raise KeyError(
                    f"{path}:{line_number} 缺少字段："
                    f"{sorted(missing)}"
                )

            rows.append(row)

    return rows


def atomic_write_jsonl(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
    ) as file:
        temporary_path = Path(file.name)

        for row in rows:
            file.write(
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
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
    ) as file:
        temporary_path = Path(file.name)

        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(output_path)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[
                self.parent[value]
            ]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root

        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def pair_sort_key(row: dict[str, Any]) -> tuple[int, Any]:
    pair_id = str(row["pair_id"])
    if pair_id.isdigit():
        return 0, int(pair_id)
    return 1, pair_id


def build_pointwise_rows(
    pairwise_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pointwise_rows: list[dict[str, Any]] = []

    for row in pairwise_rows:
        pair_id = str(row["pair_id"])
        room_id = str(row["room_id"])
        reference_path = str(
            row["reference_image_path"]
        )

        pointwise_rows.append(
            {
                "sample_id": (
                    f"layout100:{pair_id}:GOOD"
                ),
                "pair_id": pair_id,
                "room_id": room_id,
                "empty_room_image": reference_path,
                "generated_furniture_image": str(
                    row["positive_image_path"]
                ),
                "candidate_role": "GOOD",
                "label": 1,
            }
        )

        pointwise_rows.append(
            {
                "sample_id": (
                    f"layout100:{pair_id}:BAD"
                ),
                "pair_id": pair_id,
                "room_id": room_id,
                "empty_room_image": reference_path,
                "generated_furniture_image": str(
                    row["negative_image_path"]
                ),
                "candidate_role": "BAD",
                "label": 0,
            }
        )

    return pointwise_rows


def select_validation_components(
    components: list[list[int]],
    val_pairs: int,
    seed: int,
) -> set[int]:
    component_order = list(range(len(components)))
    random.Random(seed).shuffle(component_order)

    # 子集和：选择若干完整内容组件，使验证 Pair 数精确等于 val_pairs。
    choices: dict[int, list[int]] = {0: []}

    for component_index in component_order:
        component_size = len(
            components[component_index]
        )

        for current_total, selected in list(
            choices.items()
        ):
            new_total = current_total + component_size

            if (
                new_total <= val_pairs
                and new_total not in choices
            ):
                choices[new_total] = (
                    selected + [component_index]
                )

        if val_pairs in choices:
            break

    if val_pairs not in choices:
        component_sizes = sorted(
            len(component)
            for component in components
        )
        raise RuntimeError(
            "无法在不拆分重复内容组件的情况下生成精确验证集。"
            f" target={val_pairs}, "
            f"component_sizes={component_sizes}"
        )

    return set(choices[val_pairs])


def main() -> None:
    args = parse_args()

    source_manifest = (
        args.source_manifest.expanduser().resolve()
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
    )

    rows = read_jsonl(source_manifest)

    if len(rows) != args.expected_pairs:
        raise RuntimeError(
            "Pair 数量不符合预期："
            f"expected={args.expected_pairs}, "
            f"actual={len(rows)}"
        )

    pair_ids = [
        str(row["pair_id"])
        for row in rows
    ]

    duplicate_pair_ids = [
        pair_id
        for pair_id, count in Counter(
            pair_ids
        ).items()
        if count > 1
    ]

    if duplicate_pair_ids:
        raise RuntimeError(
            f"发现重复 pair_id：{duplicate_pair_ids}"
        )

    union_find = UnionFind(len(rows))
    hash_to_pair_indices: dict[str, list[int]] = (
        defaultdict(list)
    )
    row_hashes: list[dict[str, str]] = []

    for row_index, row in enumerate(rows):
        hashes: dict[str, str] = {}

        for field in IMAGE_FIELDS:
            path = resolve_image_path(
                str(row[field])
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"pair_id={row['pair_id']} "
                    f"field={field} path={path}"
                )

            digest = sha256_file(path)
            hashes[field] = digest
            hash_to_pair_indices[digest].append(
                row_index
            )

        if (
            hashes["positive_image_path"]
            == hashes["negative_image_path"]
        ):
            raise RuntimeError(
                "同一 Pair 的 GOOD 和 BAD 图片内容完全相同："
                f"pair_id={row['pair_id']}"
            )

        row_hashes.append(hashes)

    # 任意一张完全相同的图片出现在不同 Pair 时，
    # 将这些 Pair 绑定为同一组件，禁止跨 Split。
    for pair_indices in hash_to_pair_indices.values():
        first = pair_indices[0]
        for other in pair_indices[1:]:
            union_find.union(first, other)

    component_map: dict[int, list[int]] = (
        defaultdict(list)
    )

    for row_index in range(len(rows)):
        component_map[
            union_find.find(row_index)
        ].append(row_index)

    components = list(component_map.values())

    validation_component_indices = (
        select_validation_components(
            components=components,
            val_pairs=args.val_pairs,
            seed=args.seed,
        )
    )

    validation_row_indices: set[int] = set()

    for component_index in (
        validation_component_indices
    ):
        validation_row_indices.update(
            components[component_index]
        )

    train_rows = sorted(
        [
            row
            for index, row in enumerate(rows)
            if index not in validation_row_indices
        ],
        key=pair_sort_key,
    )

    val_rows = sorted(
        [
            row
            for index, row in enumerate(rows)
            if index in validation_row_indices
        ],
        key=pair_sort_key,
    )

    if len(val_rows) != args.val_pairs:
        raise AssertionError(
            f"Validation Pair 数异常：{len(val_rows)}"
        )

    if (
        len(train_rows) + len(val_rows)
        != len(rows)
    ):
        raise AssertionError("Split 样本数量不守恒")

    train_pointwise = build_pointwise_rows(
        train_rows
    )
    val_pointwise = build_pointwise_rows(
        val_rows
    )

    train_pair_ids = {
        str(row["pair_id"])
        for row in train_rows
    }
    val_pair_ids = {
        str(row["pair_id"])
        for row in val_rows
    }

    if train_pair_ids & val_pair_ids:
        raise AssertionError(
            "Train 和 Validation 存在 Pair 重叠"
        )

    train_indices = {
        index
        for index in range(len(rows))
        if index not in validation_row_indices
    }

    train_hashes = {
        digest
        for index in train_indices
        for digest in row_hashes[index].values()
    }
    val_hashes = {
        digest
        for index in validation_row_indices
        for digest in row_hashes[index].values()
    }

    cross_split_hash_overlap = (
        train_hashes & val_hashes
    )

    if cross_split_hash_overlap:
        raise AssertionError(
            "Train 和 Validation 存在图片内容重叠"
        )

    train_pairwise_path = (
        output_dir / "train_pairwise.jsonl"
    )
    val_pairwise_path = (
        output_dir / "val_pairwise.jsonl"
    )
    train_pointwise_path = (
        output_dir / "train_pointwise.jsonl"
    )
    val_pointwise_path = (
        output_dir / "val_pointwise.jsonl"
    )
    report_path = (
        output_dir / "split_report.json"
    )

    atomic_write_jsonl(
        train_rows,
        train_pairwise_path,
    )
    atomic_write_jsonl(
        val_rows,
        val_pairwise_path,
    )
    atomic_write_jsonl(
        train_pointwise,
        train_pointwise_path,
    )
    atomic_write_jsonl(
        val_pointwise,
        val_pointwise_path,
    )

    repeated_hash_groups = {
        digest: [
            str(rows[index]["pair_id"])
            for index in sorted(set(indices))
        ]
        for digest, indices
        in hash_to_pair_indices.items()
        if len(set(indices)) > 1
    }

    report = {
        "source_manifest": str(
            source_manifest
        ),
        "seed": args.seed,
        "split_policy": (
            "Pair-level split with exact-image-hash "
            "components isolated across splits"
        ),
        "expected_pairs": args.expected_pairs,
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "train_pointwise_samples": len(
            train_pointwise
        ),
        "val_pointwise_samples": len(
            val_pointwise
        ),
        "train_label_counts": dict(
            Counter(
                row["label"]
                for row in train_pointwise
            )
        ),
        "val_label_counts": dict(
            Counter(
                row["label"]
                for row in val_pointwise
            )
        ),
        "content_components": len(
            components
        ),
        "component_sizes": sorted(
            [
                len(component)
                for component in components
            ],
            reverse=True,
        ),
        "repeated_image_hash_groups": (
            repeated_hash_groups
        ),
        "cross_split_image_hash_overlap": 0,
        "train_pair_ids": [
            str(row["pair_id"])
            for row in train_rows
        ],
        "val_pair_ids": [
            str(row["pair_id"])
            for row in val_rows
        ],
        "outputs": {
            "train_pairwise": str(
                train_pairwise_path
            ),
            "val_pairwise": str(
                val_pairwise_path
            ),
            "train_pointwise": str(
                train_pointwise_path
            ),
            "val_pointwise": str(
                val_pointwise_path
            ),
        },
    }

    atomic_write_json(
        report,
        report_path,
    )

    print("Layout100 Curated Split 构建完成")
    print(f"Train Pair：{len(train_rows)}")
    print(f"Validation Pair：{len(val_rows)}")
    print(
        "Train Pointwise："
        f"{len(train_pointwise)}"
    )
    print(
        "Validation Pointwise："
        f"{len(val_pointwise)}"
    )
    print(
        "内容隔离组件："
        f"{len(components)}"
    )
    print(
        "重复图片哈希组："
        f"{len(repeated_hash_groups)}"
    )
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
