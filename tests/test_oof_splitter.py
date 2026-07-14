#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OOF 分组划分测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from preference_reward.data.oof_splitter import (
    build_oof_splits,
)


def write_jsonl(
    rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    """写入测试 JSONL。"""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取测试 JSONL。"""

    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()

            if text:
                rows.append(json.loads(text))

    return rows


def make_rows() -> List[Dict[str, Any]]:
    """生成具有多房间和双类别的测试数据。"""

    rows: List[Dict[str, Any]] = []

    for room_index in range(16):
        room_path = (
            f"data/images/empty/room_{room_index}.jpg"
        )

        # 每个房间包含 3 条样本，标签模式随房间变化，
        # 保证各分组和全局都存在足够的正负样本。
        labels = (
            [1, 1, 0]
            if room_index % 2 == 0
            else [0, 0, 1]
        )

        for candidate_index, label in enumerate(labels):
            rows.append(
                {
                    "empty_room_image": room_path,
                    "generated_furniture_image": (
                        "data/images/generated/"
                        f"room_{room_index}_"
                        f"candidate_{candidate_index}.jpg"
                    ),
                    "label": label,
                }
            )

    return rows


def test_build_oof_splits_is_group_disjoint_and_complete(
    tmp_path: Path,
) -> None:
    """外折完整覆盖，且三方集合无房间泄漏。"""

    input_manifest = tmp_path / "train.jsonl"
    output_dir = tmp_path / "oof"
    rows = make_rows()
    write_jsonl(rows, input_manifest)

    report = build_oof_splits(
        input_manifest=input_manifest,
        output_dir=output_dir,
        folds=4,
        inner_validation_ratio=0.2,
        seed=42,
        search_trials=300,
    )

    assert report["total"]["samples"] == len(rows)
    assert (
        report["outer_coverage"]["covered_samples"]
        == len(rows)
    )

    all_outer_ids: set[str] = set()
    outer_sizes: List[int] = []

    for fold_index in range(1, 5):
        fold_dir = output_dir / f"fold_{fold_index}"
        inner_train = read_jsonl(
            fold_dir / "inner_train.jsonl"
        )
        inner_val = read_jsonl(
            fold_dir / "inner_val.jsonl"
        )
        outer = read_jsonl(
            fold_dir / "outer_holdout.jsonl"
        )

        train_groups = {
            row["empty_room_image"]
            for row in inner_train
        }
        val_groups = {
            row["empty_room_image"]
            for row in inner_val
        }
        outer_groups = {
            row["empty_room_image"]
            for row in outer
        }

        assert not (train_groups & val_groups)
        assert not (train_groups & outer_groups)
        assert not (val_groups & outer_groups)

        outer_sizes.append(len(outer))

        outer_ids = {
            row["sample_id"]
            for row in outer
        }
        assert not (all_outer_ids & outer_ids)
        all_outer_ids.update(outer_ids)

        for subset in (inner_train, inner_val, outer):
            assert {
                int(row["label"])
                for row in subset
            } == {0, 1}

    assert len(all_outer_ids) == len(rows)

    # 每个房间组有 3 条样本，均衡划分的 Fold 大小差距
    # 不应超过一个完整分组。
    assert max(outer_sizes) - min(outer_sizes) <= 3


def make_singleton_group_rows() -> List[Dict[str, Any]]:
    """生成每个空房间仅对应一条样本的数据。

    该形态与当前真实训练清单一致，用于防止外折划分退化为
    3/316/114/316 之类的极端不均衡结果。
    """

    rows: List[Dict[str, Any]] = []

    for index in range(80):
        rows.append(
            {
                "empty_room_image": (
                    f"data/images/empty/room_{index}.jpg"
                ),
                "generated_furniture_image": (
                    "data/images/generated/"
                    f"candidate_{index}.jpg"
                ),
                "label": 0 if index % 3 == 0 else 1,
            }
        )

    return rows


def test_singleton_groups_are_balanced(
    tmp_path: Path,
) -> None:
    """单样本分组也必须生成大小均衡的外折。"""

    input_manifest = tmp_path / "train_singleton.jsonl"
    output_dir = tmp_path / "oof_singleton"
    rows = make_singleton_group_rows()
    write_jsonl(rows, input_manifest)

    report = build_oof_splits(
        input_manifest=input_manifest,
        output_dir=output_dir,
        folds=4,
        inner_validation_ratio=0.2,
        seed=42,
        search_trials=100,
    )

    outer_sizes = [
        int(fold["outer_holdout"]["samples"])
        for fold in report["fold_reports"]
    ]

    assert sum(outer_sizes) == len(rows)
    assert max(outer_sizes) - min(outer_sizes) <= 1

    for fold in report["fold_reports"]:
        counts = fold["outer_holdout"]["label_counts"]
        assert int(counts["0"]) > 0
        assert int(counts["1"]) > 0


def test_build_oof_splits_is_reproducible(
    tmp_path: Path,
) -> None:
    """相同输入和种子产生相同外折分配。"""

    input_manifest = tmp_path / "train.jsonl"
    output_a = tmp_path / "oof_a"
    output_b = tmp_path / "oof_b"
    write_jsonl(make_rows(), input_manifest)

    common_kwargs = {
        "input_manifest": input_manifest,
        "folds": 4,
        "inner_validation_ratio": 0.2,
        "seed": 42,
        "search_trials": 300,
    }

    build_oof_splits(
        output_dir=output_a,
        **common_kwargs,
    )
    build_oof_splits(
        output_dir=output_b,
        **common_kwargs,
    )

    assignments_a = read_jsonl(
        output_a / "oof_assignments.jsonl"
    )
    assignments_b = read_jsonl(
        output_b / "oof_assignments.jsonl"
    )

    assert assignments_a == assignments_b
