#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""通用 Pairwise Reward 评估模块。

本模块只负责评估逻辑：

1. 将 PairwiseSample 展开为正负候选评分输入；
2. 将已经完成的单候选评分重新组合成 Pairwise 结果；
3. 计算 Pairwise Accuracy 与 margin 统计；
4. 按 reference + positive 内容锚点分组计算 Bootstrap CI。

本模块不负责：

- 模型加载；
- Processor 加载；
- 模型前向；
- LoRA 加载；
- 训练或参数更新。
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from preference_reward.data.pairwise_manifest import (
    PairwiseSample,
)
from preference_reward.evaluation.classification import (
    safe_roc_auc,
)
from preference_reward.inference.scoring import (
    RewardScoringSample,
)


def file_sha256(
    path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """计算文件内容 SHA256。"""

    if block_size <= 0:
        raise ValueError(
            "block_size 必须大于 0"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(block_size)

            if not block:
                break

            hasher.update(block)

    return hasher.hexdigest()


def compute_pairwise_anchor_hash(
    reference_image: Path,
    positive_image: Path,
    hash_cache: Dict[Path, str] | None = None,
) -> str:
    """计算 reference + positive 的内容锚点。

    相同原图和相同 GOOD 图会得到相同 anchor_hash，
    即使文件名、pair_id 或目录不同。
    """

    if hash_cache is None:
        hash_cache = {}

    def cached_hash(path: Path) -> str:
        resolved = path.resolve()

        if resolved not in hash_cache:
            hash_cache[resolved] = file_sha256(
                resolved
            )

        return hash_cache[resolved]

    reference_hash = cached_hash(
        reference_image
    )
    positive_hash = cached_hash(
        positive_image
    )

    hasher = hashlib.sha256()
    hasher.update(
        reference_hash.encode("ascii")
    )
    hasher.update(b"\0PAIR_ANCHOR\0")
    hasher.update(
        positive_hash.encode("ascii")
    )

    return hasher.hexdigest()


def build_pairwise_scoring_samples(
    pairwise_samples: Sequence[PairwiseSample],
) -> List[RewardScoringSample]:
    """将每个 Pair 展开为 positive 和 negative 两条评分输入。"""

    if not pairwise_samples:
        raise ValueError(
            "pairwise_samples 不能为空"
        )

    scoring_samples: List[
        RewardScoringSample
    ] = []

    seen_sample_ids: set[str] = set()

    for pair in pairwise_samples:
        positive_sample_id = (
            f"{pair.pair_id}::positive"
        )
        negative_sample_id = (
            f"{pair.pair_id}::negative"
        )

        for sample_id in (
            positive_sample_id,
            negative_sample_id,
        ):
            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"评分 sample_id 重复："
                    f"{sample_id}"
                )

            seen_sample_ids.add(sample_id)

        common_metadata = {
            "pair_id": pair.pair_id,
            **dict(pair.metadata),
        }

        scoring_samples.append(
            RewardScoringSample(
                sample_id=positive_sample_id,
                empty_room_image=(
                    pair.reference_image
                ),
                generated_furniture_image=(
                    pair.positive_image
                ),
                metadata={
                    **common_metadata,
                    "role": "positive",
                },
            )
        )

        scoring_samples.append(
            RewardScoringSample(
                sample_id=negative_sample_id,
                empty_room_image=(
                    pair.reference_image
                ),
                generated_furniture_image=(
                    pair.negative_image
                ),
                metadata={
                    **common_metadata,
                    "role": "negative",
                },
            )
        )

    return scoring_samples


def index_scoring_rows(
    scoring_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    """按 sample_id 索引单候选评分结果。"""

    if not scoring_rows:
        raise ValueError(
            "scoring_rows 不能为空"
        )

    result: Dict[
        str,
        Mapping[str, Any],
    ] = {}

    required_fields = {
        "sample_id",
        "reward_score",
        "p_like_raw",
    }

    for index, row in enumerate(
        scoring_rows,
        start=1,
    ):
        missing_fields = (
            required_fields - set(row)
        )

        if missing_fields:
            raise KeyError(
                f"第 {index} 条评分结果缺少字段："
                f"{sorted(missing_fields)}"
            )

        sample_id = str(
            row["sample_id"]
        )

        if sample_id in result:
            raise ValueError(
                f"评分 sample_id 重复："
                f"{sample_id}"
            )

        result[sample_id] = row

    return result


def assemble_pairwise_rows(
    pairwise_samples: Sequence[PairwiseSample],
    scoring_rows: Sequence[Mapping[str, Any]],
    tie_tolerance: float = 1e-8,
) -> List[Dict[str, Any]]:
    """将单候选评分组合成 Pairwise 结果。"""

    if not pairwise_samples:
        raise ValueError(
            "pairwise_samples 不能为空"
        )

    if tie_tolerance < 0:
        raise ValueError(
            "tie_tolerance 不能小于 0"
        )

    score_map = index_scoring_rows(
        scoring_rows
    )

    hash_cache: Dict[Path, str] = {}
    pair_rows: List[Dict[str, Any]] = []

    for pair in pairwise_samples:
        positive_id = (
            f"{pair.pair_id}::positive"
        )
        negative_id = (
            f"{pair.pair_id}::negative"
        )

        if positive_id not in score_map:
            raise KeyError(
                f"缺少正样本评分：{positive_id}"
            )

        if negative_id not in score_map:
            raise KeyError(
                f"缺少负样本评分：{negative_id}"
            )

        positive_row = score_map[
            positive_id
        ]
        negative_row = score_map[
            negative_id
        ]

        positive_score = float(
            positive_row["reward_score"]
        )
        negative_score = float(
            negative_row["reward_score"]
        )

        if not np.isfinite(positive_score):
            raise ValueError(
                f"{positive_id} reward_score "
                "不是有限数"
            )

        if not np.isfinite(negative_score):
            raise ValueError(
                f"{negative_id} reward_score "
                "不是有限数"
            )

        margin = (
            positive_score
            - negative_score
        )

        is_tie = (
            abs(margin) <= tie_tolerance
        )
        is_correct = (
            margin > tie_tolerance
        )
        is_incorrect = (
            margin < -tie_tolerance
        )

        anchor_hash = (
            compute_pairwise_anchor_hash(
                reference_image=(
                    pair.reference_image
                ),
                positive_image=(
                    pair.positive_image
                ),
                hash_cache=hash_cache,
            )
        )

        pair_rows.append(
            {
                "pair_id": pair.pair_id,
                "anchor_hash": anchor_hash,
                "reference_image_path": str(
                    pair.reference_image
                ),
                "positive_image_path": str(
                    pair.positive_image
                ),
                "negative_image_path": str(
                    pair.negative_image
                ),
                "positive_reward_score": (
                    positive_score
                ),
                "negative_reward_score": (
                    negative_score
                ),
                "pairwise_margin": float(
                    margin
                ),
                "is_correct": bool(
                    is_correct
                ),
                "is_incorrect": bool(
                    is_incorrect
                ),
                "is_tie": bool(
                    is_tie
                ),
                "positive_p_like_raw": float(
                    positive_row["p_like_raw"]
                ),
                "negative_p_like_raw": float(
                    negative_row["p_like_raw"]
                ),
                "metadata": dict(
                    pair.metadata
                ),
            }
        )

    return pair_rows


def compute_pairwise_metrics(
    pair_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """计算 Pairwise 排序与分数分布指标。"""

    if not pair_rows:
        raise ValueError(
            "pair_rows 不能为空"
        )

    margins = np.asarray(
        [
            float(row["pairwise_margin"])
            for row in pair_rows
        ],
        dtype=np.float64,
    )

    positive_scores = np.asarray(
        [
            float(
                row[
                    "positive_reward_score"
                ]
            )
            for row in pair_rows
        ],
        dtype=np.float64,
    )

    negative_scores = np.asarray(
        [
            float(
                row[
                    "negative_reward_score"
                ]
            )
            for row in pair_rows
        ],
        dtype=np.float64,
    )

    correct = np.asarray(
        [
            bool(row["is_correct"])
            for row in pair_rows
        ],
        dtype=bool,
    )

    incorrect = np.asarray(
        [
            bool(row["is_incorrect"])
            for row in pair_rows
        ],
        dtype=bool,
    )

    ties = np.asarray(
        [
            bool(row["is_tie"])
            for row in pair_rows
        ],
        dtype=bool,
    )

    if not np.all(
        correct.astype(np.int64)
        + incorrect.astype(np.int64)
        + ties.astype(np.int64)
        == 1
    ):
        raise ValueError(
            "每个 Pair 必须且只能属于"
            " correct / incorrect / tie 之一"
        )

    pointwise_labels = np.concatenate(
        [
            np.ones(
                len(positive_scores),
                dtype=np.int64,
            ),
            np.zeros(
                len(negative_scores),
                dtype=np.int64,
            ),
        ]
    )

    pointwise_scores = np.concatenate(
        [
            positive_scores,
            negative_scores,
        ]
    )

    unique_anchors = {
        str(row["anchor_hash"])
        for row in pair_rows
    }

    return {
        "num_pairs": int(
            len(pair_rows)
        ),
        "num_correct": int(
            correct.sum()
        ),
        "num_incorrect": int(
            incorrect.sum()
        ),
        "num_ties": int(
            ties.sum()
        ),
        "pairwise_accuracy_strict": float(
            correct.mean()
        ),
        "pairwise_accuracy_tie_half": float(
            (
                correct.astype(np.float64)
                + 0.5
                * ties.astype(np.float64)
            ).mean()
        ),
        "mean_pairwise_margin": float(
            margins.mean()
        ),
        "median_pairwise_margin": float(
            np.median(margins)
        ),
        "std_pairwise_margin": float(
            margins.std()
        ),
        "minimum_pairwise_margin": float(
            margins.min()
        ),
        "maximum_pairwise_margin": float(
            margins.max()
        ),
        "mean_positive_reward_score": float(
            positive_scores.mean()
        ),
        "mean_negative_reward_score": float(
            negative_scores.mean()
        ),
        "mean_positive_minus_negative": float(
            positive_scores.mean()
            - negative_scores.mean()
        ),
        "pointwise_roc_auc": safe_roc_auc(
            pointwise_labels,
            pointwise_scores,
        ),
        "num_unique_anchors": int(
            len(unique_anchors)
        ),
    }


def grouped_bootstrap_pairwise_accuracy(
    pair_rows: Sequence[Mapping[str, Any]],
    iterations: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """按 anchor_hash 分组计算 Pairwise Accuracy 置信区间。

    每轮 Bootstrap：

    1. 从唯一 anchor 中有放回抽样；
    2. 被抽中的 anchor 保留其全部 Pair；
    3. 计算 strict pairwise accuracy。

    这样内容重复的锚点不会被当成完全独立样本。
    """

    if not pair_rows:
        raise ValueError(
            "pair_rows 不能为空"
        )

    if iterations <= 0:
        raise ValueError(
            "iterations 必须大于 0"
        )

    grouped: Dict[
        str,
        List[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in pair_rows:
        grouped[
            str(row["anchor_hash"])
        ].append(row)

    anchor_ids = sorted(grouped)

    if not anchor_ids:
        raise RuntimeError(
            "没有可用于 Bootstrap 的锚点"
        )

    rng = np.random.default_rng(seed)

    bootstrap_values = np.empty(
        iterations,
        dtype=np.float64,
    )

    for iteration in range(iterations):
        sampled_anchor_ids = rng.choice(
            anchor_ids,
            size=len(anchor_ids),
            replace=True,
        )

        sampled_correct: List[float] = []

        for anchor_id in sampled_anchor_ids:
            for row in grouped[
                str(anchor_id)
            ]:
                sampled_correct.append(
                    float(
                        bool(row["is_correct"])
                    )
                )

        bootstrap_values[
            iteration
        ] = float(
            np.mean(sampled_correct)
        )

    return {
        "bootstrap_iterations": int(
            iterations
        ),
        "bootstrap_seed": int(seed),
        "bootstrap_num_groups": int(
            len(anchor_ids)
        ),
        "pairwise_accuracy_bootstrap_mean": float(
            bootstrap_values.mean()
        ),
        "pairwise_accuracy_bootstrap_std": float(
            bootstrap_values.std()
        ),
        "pairwise_accuracy_ci95_lower": float(
            np.percentile(
                bootstrap_values,
                2.5,
            )
        ),
        "pairwise_accuracy_ci95_upper": float(
            np.percentile(
                bootstrap_values,
                97.5,
            )
        ),
    }


def evaluate_pairwise_scores(
    pairwise_samples: Sequence[PairwiseSample],
    scoring_rows: Sequence[Mapping[str, Any]],
    tie_tolerance: float = 1e-8,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 42,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """组合评分并返回完整 Pairwise 指标与逐 Pair 结果。"""

    pair_rows = assemble_pairwise_rows(
        pairwise_samples=pairwise_samples,
        scoring_rows=scoring_rows,
        tie_tolerance=tie_tolerance,
    )

    metrics = compute_pairwise_metrics(
        pair_rows
    )

    metrics.update(
        grouped_bootstrap_pairwise_accuracy(
            pair_rows=pair_rows,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
    )

    return metrics, pair_rows
