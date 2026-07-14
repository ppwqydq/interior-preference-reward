#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""按空房间分组构建 OOF 外折与折内训练/验证清单。

约束：
1. 仅使用原训练集构建 OOF；固定验证集不参与。
2. 以 ``empty_room_image`` 为分组键，避免同一空间跨集合泄漏。
3. 每条样本生成稳定 ``sample_id``，便于合并折外预测。
4. 每个外折只用于最终折外预测，不用于早停或选择最佳 Epoch。
5. 外折之外的数据再分为 ``inner_train`` 和 ``inner_val``。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)
from preference_reward.data.splitter import (
    group_rows,
    label_counts,
    positive_rate,
    read_jsonl,
    search_validation_groups,
)


def stable_sample_id(row: Mapping[str, Any]) -> str:
    """根据两张图片路径生成稳定样本 ID。"""

    payload = (
        str(row["empty_room_image"]).strip()
        + "\n"
        + str(row["generated_furniture_image"]).strip()
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def add_and_validate_sample_ids(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """补充稳定 sample_id，并校验样本唯一性。"""

    normalized_rows: List[Dict[str, Any]] = []
    seen_sample_ids: Dict[str, tuple[str, str]] = {}
    seen_pairs: set[tuple[str, str]] = set()

    for row in rows:
        normalized = dict(row)

        pair = (
            str(normalized["empty_room_image"]),
            str(normalized["generated_furniture_image"]),
        )

        if pair in seen_pairs:
            raise ValueError(
                "输入清单存在重复图片对："
                f"empty={pair[0]} generated={pair[1]}"
            )

        seen_pairs.add(pair)

        generated_id = stable_sample_id(normalized)
        existing_id = str(
            normalized.get("sample_id", "")
        ).strip()
        sample_id = existing_id or generated_id

        if sample_id in seen_sample_ids:
            previous_pair = seen_sample_ids[sample_id]
            raise ValueError(
                "sample_id 重复："
                f"{sample_id}，"
                f"已有图片对={previous_pair}，"
                f"当前图片对={pair}"
            )

        seen_sample_ids[sample_id] = pair
        normalized["sample_id"] = sample_id
        normalized["label"] = int(normalized["label"])
        normalized_rows.append(normalized)

    return normalized_rows


def _group_statistics(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Dict[str, int]]:
    """统计每个空房间分组的样本数和正样本数。"""

    statistics: Dict[str, Dict[str, int]] = {}

    for key, items in groups.items():
        statistics[key] = {
            "samples": len(items),
            "positives": sum(
                int(item["label"])
                for item in items
            ),
        }

    return statistics


def _assignment_score(
    fold_group_keys: Sequence[Sequence[str]],
    statistics: Mapping[str, Mapping[str, int]],
    target_samples_per_fold: float,
    overall_positive_rate: float,
) -> float:
    """评价外折划分质量，分数越低越好。"""

    score = 0.0

    for keys in fold_group_keys:
        samples = sum(
            int(statistics[key]["samples"])
            for key in keys
        )
        positives = sum(
            int(statistics[key]["positives"])
            for key in keys
        )

        if samples <= 0:
            return float("inf")

        negatives = samples - positives

        if positives <= 0 or negatives <= 0:
            return float("inf")

        fold_positive_rate = positives / samples

        score += (
            abs(samples - target_samples_per_fold)
            / target_samples_per_fold
        )
        score += 2.0 * abs(
            fold_positive_rate
            - overall_positive_rate
        )

    return score / len(fold_group_keys)


def _partial_balance_score(
    fold_samples: Sequence[int],
    fold_positives: Sequence[int],
    target_samples_per_fold: float,
    target_positives_per_fold: float,
    target_negatives_per_fold: float,
) -> float:
    """评价当前部分分配的全局均衡程度。

    使用平方误差而不是单个 Fold 的绝对误差。这样继续向已经
    较大的 Fold 添加样本会受到更强惩罚，可避免出现一个 Fold
    持续吸收样本、其他 Fold 几乎为空的退化结果。
    """

    score = 0.0

    for samples, positives in zip(
        fold_samples,
        fold_positives,
    ):
        negatives = samples - positives

        sample_error = (
            (samples - target_samples_per_fold)
            / max(target_samples_per_fold, 1.0)
        )
        positive_error = (
            (positives - target_positives_per_fold)
            / max(target_positives_per_fold, 1.0)
        )
        negative_error = (
            (negatives - target_negatives_per_fold)
            / max(target_negatives_per_fold, 1.0)
        )

        score += sample_error ** 2
        score += positive_error ** 2
        score += negative_error ** 2

    return score


def search_outer_folds(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    folds: int,
    seed: int,
    search_trials: int,
) -> List[set[str]]:
    """通过确定性多次随机贪心搜索构建均衡外层 Group Fold。

    优化目标同时约束每折：
    - 总样本数；
    - 正样本数；
    - 负样本数。

    每次分配一个完整空房间分组，任何分组都不会跨 Fold。
    """

    if folds < 2:
        raise ValueError("folds 必须至少为 2")

    if search_trials <= 0:
        raise ValueError("search_trials 必须大于 0")

    group_keys = sorted(groups)

    if len(group_keys) < folds:
        raise RuntimeError(
            "空房间分组数少于 Fold 数："
            f"groups={len(group_keys)} folds={folds}"
        )

    statistics = _group_statistics(groups)
    total_samples = sum(
        value["samples"]
        for value in statistics.values()
    )
    total_positives = sum(
        value["positives"]
        for value in statistics.values()
    )
    total_negatives = (
        total_samples - total_positives
    )
    overall_positive_rate = (
        total_positives / total_samples
    )

    target_samples_per_fold = (
        total_samples / folds
    )
    target_positives_per_fold = (
        total_positives / folds
    )
    target_negatives_per_fold = (
        total_negatives / folds
    )

    best_assignment: List[set[str]] | None = None
    best_score: float | None = None

    for trial in range(search_trials):
        rng = random.Random(seed + trial)
        ordered_keys = list(group_keys)
        rng.shuffle(ordered_keys)

        # 大分组和类别偏斜更明显的分组优先分配。排序稳定，
        # 同优先级分组仍保留上面的随机顺序。
        ordered_keys.sort(
            key=lambda key: (
                int(statistics[key]["samples"]),
                abs(
                    int(statistics[key]["positives"])
                    - int(statistics[key]["samples"])
                    * overall_positive_rate
                ),
            ),
            reverse=True,
        )

        fold_keys: List[List[str]] = [
            [] for _ in range(folds)
        ]
        fold_samples = [0 for _ in range(folds)]
        fold_positives = [0 for _ in range(folds)]

        for key in ordered_keys:
            group_samples = int(
                statistics[key]["samples"]
            )
            group_positives = int(
                statistics[key]["positives"]
            )

            candidate_indices = list(range(folds))
            rng.shuffle(candidate_indices)

            best_fold_index: int | None = None
            best_local_score: float | None = None

            for fold_index in candidate_indices:
                candidate_samples = list(
                    fold_samples
                )
                candidate_positives = list(
                    fold_positives
                )
                candidate_samples[fold_index] += (
                    group_samples
                )
                candidate_positives[fold_index] += (
                    group_positives
                )

                local_score = _partial_balance_score(
                    fold_samples=candidate_samples,
                    fold_positives=(
                        candidate_positives
                    ),
                    target_samples_per_fold=(
                        target_samples_per_fold
                    ),
                    target_positives_per_fold=(
                        target_positives_per_fold
                    ),
                    target_negatives_per_fold=(
                        target_negatives_per_fold
                    ),
                )

                if (
                    best_local_score is None
                    or local_score < best_local_score
                ):
                    best_local_score = local_score
                    best_fold_index = fold_index

            if best_fold_index is None:
                raise RuntimeError(
                    "外折分配内部错误：未找到候选 Fold"
                )

            fold_keys[best_fold_index].append(key)
            fold_samples[best_fold_index] += (
                group_samples
            )
            fold_positives[best_fold_index] += (
                group_positives
            )

        score = _assignment_score(
            fold_group_keys=fold_keys,
            statistics=statistics,
            target_samples_per_fold=(
                target_samples_per_fold
            ),
            overall_positive_rate=(
                overall_positive_rate
            ),
        )

        if not (score < float("inf")):
            continue

        candidate_sets = [
            set(keys)
            for keys in fold_keys
        ]

        # 每个外折的补集也必须同时包含正负样本。
        valid = True
        all_group_keys = set(group_keys)

        for outer_keys in candidate_sets:
            remaining_keys = (
                all_group_keys - outer_keys
            )
            remaining_labels = {
                int(row["label"])
                for group_key in remaining_keys
                for row in groups[group_key]
            }

            if remaining_labels != {0, 1}:
                valid = False
                break

        if not valid:
            continue

        if best_score is None or score < best_score:
            best_score = score
            best_assignment = candidate_sets

    if best_assignment is None:
        raise RuntimeError(
            "未找到每折及其补集都包含正负样本的有效 OOF 划分"
        )

    # 防止退化成极端不均衡 Fold。理论上均衡贪心的样本数差距
    # 不应显著超过最大单个分组的大小。预留 2 倍容差以兼容
    # 分组大小和类别比例同时受约束的场景。
    outer_sizes = [
        sum(
            int(statistics[key]["samples"])
            for key in fold_keys
        )
        for fold_keys in best_assignment
    ]
    largest_group_size = max(
        int(value["samples"])
        for value in statistics.values()
    )
    allowed_size_gap = max(
        1,
        2 * largest_group_size,
    )

    if max(outer_sizes) - min(outer_sizes) > (
        allowed_size_gap
    ):
        raise RuntimeError(
            "外折样本数严重不均衡："
            f"sizes={outer_sizes}，"
            f"allowed_gap={allowed_size_gap}"
        )

    return best_assignment

def _sorted_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """固定排序，保证相同输入与种子产生相同输出。"""

    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row["empty_room_image"]),
            str(row["generated_furniture_image"]),
            int(row["label"]),
            str(row["sample_id"]),
        ),
    )


def _manifest_summary(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """生成单个清单的样本与分组统计。"""

    labels = Counter(
        int(row["label"])
        for row in rows
    )
    groups = {
        str(row["empty_room_image"])
        for row in rows
    }

    return {
        "samples": len(rows),
        "empty_room_groups": len(groups),
        "label_counts": {
            "0": int(labels.get(0, 0)),
            "1": int(labels.get(1, 0)),
        },
        "positive_rate": (
            sum(int(row["label"]) for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
    }


def _validate_three_way_split(
    inner_train_rows: Sequence[Mapping[str, Any]],
    inner_val_rows: Sequence[Mapping[str, Any]],
    outer_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """校验 Inner Train、Inner Val、Outer Holdout 无样本或分组重叠。"""

    row_sets = {
        "inner_train": {
            str(row["sample_id"])
            for row in inner_train_rows
        },
        "inner_val": {
            str(row["sample_id"])
            for row in inner_val_rows
        },
        "outer_holdout": {
            str(row["sample_id"])
            for row in outer_rows
        },
    }
    group_sets = {
        "inner_train": {
            str(row["empty_room_image"])
            for row in inner_train_rows
        },
        "inner_val": {
            str(row["empty_room_image"])
            for row in inner_val_rows
        },
        "outer_holdout": {
            str(row["empty_room_image"])
            for row in outer_rows
        },
    }

    pair_names = [
        ("inner_train", "inner_val"),
        ("inner_train", "outer_holdout"),
        ("inner_val", "outer_holdout"),
    ]

    report: Dict[str, int] = {}

    for left, right in pair_names:
        sample_overlap = len(
            row_sets[left] & row_sets[right]
        )
        group_overlap = len(
            group_sets[left] & group_sets[right]
        )
        report[
            f"sample_overlap_{left}_{right}"
        ] = sample_overlap
        report[
            f"group_overlap_{left}_{right}"
        ] = group_overlap

        if sample_overlap or group_overlap:
            raise RuntimeError(
                "OOF 三方划分存在重叠："
                f"{left} vs {right}，"
                f"sample_overlap={sample_overlap}，"
                f"group_overlap={group_overlap}"
            )

    return report


def build_oof_splits(
    input_manifest: Path,
    output_dir: Path,
    folds: int = 4,
    inner_validation_ratio: float = 0.2,
    seed: int = 42,
    search_trials: int = 5000,
) -> Dict[str, Any]:
    """构建完整 OOF 数据划分并写入磁盘。"""

    if not 0.0 < inner_validation_ratio < 1.0:
        raise ValueError(
            "inner_validation_ratio 必须位于 0 和 1 之间"
        )

    rows = add_and_validate_sample_ids(
        read_jsonl(input_manifest)
    )
    groups = group_rows(rows)

    outer_fold_group_keys = search_outer_folds(
        groups=groups,
        folds=folds,
        seed=seed,
        search_trials=search_trials,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_group_keys = set(groups)
    all_sample_ids = {
        str(row["sample_id"])
        for row in rows
    }
    covered_outer_sample_ids: set[str] = set()
    assignments: List[Dict[str, Any]] = []
    fold_reports: List[Dict[str, Any]] = []

    for fold_index, outer_group_keys in enumerate(
        outer_fold_group_keys,
        start=1,
    ):
        remaining_group_keys = (
            all_group_keys - outer_group_keys
        )
        remaining_groups = {
            key: groups[key]
            for key in remaining_group_keys
        }
        remaining_rows = [
            row
            for key in remaining_group_keys
            for row in groups[key]
        ]

        inner_val_group_keys = (
            search_validation_groups(
                groups=remaining_groups,
                total_rows=len(remaining_rows),
                validation_ratio=(
                    inner_validation_ratio
                ),
                seed=seed + 10_000 + fold_index,
                search_trials=search_trials,
            )
        )

        inner_train_rows = _sorted_rows([
            row
            for key in remaining_group_keys
            if key not in inner_val_group_keys
            for row in groups[key]
        ])
        inner_val_rows = _sorted_rows([
            row
            for key in inner_val_group_keys
            for row in groups[key]
        ])
        outer_rows = _sorted_rows([
            row
            for key in outer_group_keys
            for row in groups[key]
        ])

        for subset_name, subset_rows in (
            ("inner_train", inner_train_rows),
            ("inner_val", inner_val_rows),
            ("outer_holdout", outer_rows),
        ):
            if {
                int(row["label"])
                for row in subset_rows
            } != {0, 1}:
                raise RuntimeError(
                    f"Fold {fold_index} 的 {subset_name} "
                    "没有同时包含正负样本"
                )

        overlap_report = _validate_three_way_split(
            inner_train_rows=inner_train_rows,
            inner_val_rows=inner_val_rows,
            outer_rows=outer_rows,
        )

        outer_sample_ids = {
            str(row["sample_id"])
            for row in outer_rows
        }
        duplicate_outer_ids = (
            covered_outer_sample_ids
            & outer_sample_ids
        )

        if duplicate_outer_ids:
            raise RuntimeError(
                "样本被分配到多个 Outer Fold："
                f"{sorted(duplicate_outer_ids)[:10]}"
            )

        covered_outer_sample_ids.update(
            outer_sample_ids
        )

        fold_dir = (
            output_dir
            / f"fold_{fold_index}"
        )
        inner_train_path = (
            fold_dir / "inner_train.jsonl"
        )
        inner_val_path = (
            fold_dir / "inner_val.jsonl"
        )
        outer_path = (
            fold_dir / "outer_holdout.jsonl"
        )

        write_jsonl_atomic(
            inner_train_rows,
            inner_train_path,
        )
        write_jsonl_atomic(
            inner_val_rows,
            inner_val_path,
        )
        write_jsonl_atomic(
            outer_rows,
            outer_path,
        )

        for row in outer_rows:
            assignments.append(
                {
                    "sample_id": row["sample_id"],
                    "outer_fold": fold_index,
                    "empty_room_image": (
                        row["empty_room_image"]
                    ),
                    "generated_furniture_image": (
                        row[
                            "generated_furniture_image"
                        ]
                    ),
                    "label": int(row["label"]),
                }
            )

        fold_reports.append(
            {
                "fold": fold_index,
                "paths": {
                    "inner_train": str(
                        inner_train_path.resolve()
                    ),
                    "inner_val": str(
                        inner_val_path.resolve()
                    ),
                    "outer_holdout": str(
                        outer_path.resolve()
                    ),
                },
                "inner_train": _manifest_summary(
                    inner_train_rows
                ),
                "inner_val": _manifest_summary(
                    inner_val_rows
                ),
                "outer_holdout": _manifest_summary(
                    outer_rows
                ),
                "overlap": overlap_report,
            }
        )

    if covered_outer_sample_ids != all_sample_ids:
        missing = sorted(
            all_sample_ids
            - covered_outer_sample_ids
        )
        unexpected = sorted(
            covered_outer_sample_ids
            - all_sample_ids
        )
        raise RuntimeError(
            "Outer Fold 没有完整且唯一地覆盖输入训练集："
            f"missing={missing[:10]}，"
            f"unexpected={unexpected[:10]}"
        )

    assignments = sorted(
        assignments,
        key=lambda row: (
            int(row["outer_fold"]),
            str(row["sample_id"]),
        ),
    )
    assignments_path = (
        output_dir / "oof_assignments.jsonl"
    )
    write_jsonl_atomic(
        assignments,
        assignments_path,
    )

    report = {
        "input_manifest": str(
            input_manifest.resolve()
        ),
        "output_dir": str(output_dir.resolve()),
        "seed": seed,
        "folds": folds,
        "inner_validation_ratio": (
            inner_validation_ratio
        ),
        "search_trials": search_trials,
        "group_key": "empty_room_image",
        "sample_id_policy": (
            "保留已有 sample_id；缺失时使用 "
            "SHA256(empty_room_image + newline + "
            "generated_furniture_image)"
        ),
        "total": {
            "samples": len(rows),
            "empty_room_groups": len(groups),
            "label_counts": label_counts(rows),
            "positive_rate": positive_rate(rows),
        },
        "outer_coverage": {
            "expected_samples": len(all_sample_ids),
            "covered_samples": len(
                covered_outer_sample_ids
            ),
            "missing_samples": 0,
            "duplicate_samples": 0,
        },
        "assignments_path": str(
            assignments_path.resolve()
        ),
        "fold_reports": fold_reports,
        "policy": {
            "fixed_validation_set_used": False,
            "outer_holdout_used_for_training": False,
            "outer_holdout_used_for_early_stopping": False,
            "inner_val_used_for_early_stopping": True,
        },
    }

    write_json_atomic(
        report,
        output_dir / "oof_split_report.json",
    )

    return report
