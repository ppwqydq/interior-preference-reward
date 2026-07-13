#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""通用 Pairwise 图片偏好清单读取模块。

标准样本结构：

    reference image
    positive candidate
    negative candidate

默认字段适配当前布局测试集，同时允许其他数据集通过
PairwiseFieldMap 指定不同字段名。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class PairwiseFieldMap:
    """JSONL 字段名映射。"""

    pair_id: str = "pair_id"
    reference_image: str = "reference_image_path"
    positive_image: str = "positive_image_path"
    negative_image: str = "negative_image_path"


@dataclass(frozen=True)
class PairwiseSample:
    """一条标准 Pairwise 图片偏好样本。"""

    pair_id: str
    reference_image: Path
    positive_image: Path
    negative_image: Path
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


def resolve_pairwise_image_path(
    value: str | Path,
    project_root: Path,
) -> Path:
    """解析绝对路径或项目相对路径。"""

    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        project_root / path
    ).resolve()


def read_pairwise_manifest(
    manifest_path: Path,
    project_root: Path,
    fields: PairwiseFieldMap | None = None,
    validate_image_paths: bool = True,
) -> List[PairwiseSample]:
    """读取并校验 Pairwise JSONL 清单。"""

    manifest_path = (
        manifest_path
        .expanduser()
        .resolve()
    )
    project_root = (
        project_root
        .expanduser()
        .resolve()
    )

    if fields is None:
        fields = PairwiseFieldMap()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Pairwise 清单不存在：{manifest_path}"
        )

    required_fields = {
        fields.pair_id,
        fields.reference_image,
        fields.positive_image,
        fields.negative_image,
    }

    samples: List[PairwiseSample] = []
    seen_pair_ids: set[str] = set()
    missing_paths: List[str] = []

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "JSON 解析失败："
                    f"{manifest_path}:{line_number}"
                ) from exc

            if not isinstance(row, dict):
                raise TypeError(
                    "Pairwise 样本必须是 JSON 对象："
                    f"{manifest_path}:{line_number}"
                )

            missing_fields = (
                required_fields - set(row)
            )

            if missing_fields:
                raise KeyError(
                    f"{manifest_path}:{line_number} "
                    f"缺少字段："
                    f"{sorted(missing_fields)}"
                )

            pair_id = str(
                row[fields.pair_id]
            ).strip()

            if not pair_id:
                raise ValueError(
                    f"{manifest_path}:{line_number} "
                    "pair_id 不能为空"
                )

            if pair_id in seen_pair_ids:
                raise ValueError(
                    f"Pair ID 重复：{pair_id}"
                )

            seen_pair_ids.add(pair_id)

            reference_image = (
                resolve_pairwise_image_path(
                    row[fields.reference_image],
                    project_root,
                )
            )
            positive_image = (
                resolve_pairwise_image_path(
                    row[fields.positive_image],
                    project_root,
                )
            )
            negative_image = (
                resolve_pairwise_image_path(
                    row[fields.negative_image],
                    project_root,
                )
            )

            if validate_image_paths:
                for role, image_path in (
                    (
                        "reference",
                        reference_image,
                    ),
                    (
                        "positive",
                        positive_image,
                    ),
                    (
                        "negative",
                        negative_image,
                    ),
                ):
                    if not image_path.is_file():
                        missing_paths.append(
                            f"pair_id={pair_id} "
                            f"role={role} "
                            f"path={image_path}"
                        )

            metadata: Dict[str, Any] = {
                key: value
                for key, value in row.items()
                if key not in required_fields
            }

            samples.append(
                PairwiseSample(
                    pair_id=pair_id,
                    reference_image=reference_image,
                    positive_image=positive_image,
                    negative_image=negative_image,
                    metadata=metadata,
                )
            )

    if not samples:
        raise RuntimeError(
            f"Pairwise 清单为空：{manifest_path}"
        )

    if missing_paths:
        preview = "\n".join(
            f"  - {item}"
            for item in missing_paths[:20]
        )

        raise FileNotFoundError(
            f"共有 {len(missing_paths)} 个"
            " Pairwise 图片路径不存在：\n"
            f"{preview}"
        )

    return samples
