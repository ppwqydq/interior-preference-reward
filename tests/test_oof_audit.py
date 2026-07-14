#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OOF 人工审计集选择逻辑测试。"""

from __future__ import annotations

import pytest

from preference_reward.evaluation.oof_audit import (
    select_balanced_audit_samples,
)


def make_rows():
    """生成 2 Fold × 2 Label × 4 条数据。"""

    rows = []

    for fold in (1, 2):
        for label in (0, 1):
            for index, confidence in enumerate(
                [0.4, 0.1, 0.3, 0.2],
                start=1,
            ):
                sample_id = (
                    f"f{fold}_l{label}_{index}"
                )
                p_like = (
                    confidence
                    if label == 1
                    else 1.0 - confidence
                )

                rows.append(
                    {
                        "sample_id": sample_id,
                        "fold": fold,
                        "label": label,
                        "label_confidence": (
                            confidence
                        ),
                        "conflict_score": (
                            1.0 - confidence
                        ),
                        "reward_score": 0.0,
                        "p_like_raw": p_like,
                        "p_like_prior_corrected": (
                            p_like
                        ),
                        "empty_room_image": (
                            f"/tmp/{sample_id}_e.png"
                        ),
                        "generated_furniture_image": (
                            f"/tmp/{sample_id}_g.png"
                        ),
                    }
                )

    return rows


def test_balanced_selection():
    """每个 Fold、每个标签应选相同数量。"""

    selected = (
        select_balanced_audit_samples(
            predictions=make_rows(),
            per_fold_per_label=2,
        )
    )

    assert len(selected) == 8

    groups = {}

    for row in selected:
        key = (
            row["fold"],
            row["label"],
        )
        groups.setdefault(
            key,
            [],
        ).append(
            row["label_confidence"]
        )

    assert set(groups) == {
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    }

    for confidences in groups.values():
        assert confidences == [
            pytest.approx(0.1),
            pytest.approx(0.2),
        ]


def test_duplicate_id_rejected():
    """重复 sample_id 必须拒绝。"""

    rows = make_rows()
    rows.append(dict(rows[0]))

    with pytest.raises(
        RuntimeError,
        match="sample_id 重复",
    ):
        select_balanced_audit_samples(
            rows,
            per_fold_per_label=1,
        )


def test_insufficient_group_rejected():
    """组内样本不足必须明确报错。"""

    rows = make_rows()

    with pytest.raises(
        RuntimeError,
        match="候选样本不足",
    ):
        select_balanced_audit_samples(
            rows,
            per_fold_per_label=5,
        )
