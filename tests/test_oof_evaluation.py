#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OOF 预测汇总模块测试。"""

from __future__ import annotations

import pytest

from preference_reward.evaluation.oof import (
    build_confidence_report,
    compute_oof_metrics,
    enrich_oof_predictions,
    validate_oof_coverage,
)


def make_predictions():
    """构造最小双类别 OOF 预测。"""

    return [
        {
            "sample_id": "a",
            "fold": 1,
            "label": 1,
            "reward_score": 2.0,
            "p_like_raw": 0.80,
            "p_like_prior_corrected": 0.90,
        },
        {
            "sample_id": "b",
            "fold": 1,
            "label": 0,
            "reward_score": -1.0,
            "p_like_raw": 0.30,
            "p_like_prior_corrected": 0.20,
        },
        {
            "sample_id": "c",
            "fold": 2,
            "label": 1,
            "reward_score": -0.5,
            "p_like_raw": 0.40,
            "p_like_prior_corrected": 0.45,
        },
        {
            "sample_id": "d",
            "fold": 2,
            "label": 0,
            "reward_score": 0.7,
            "p_like_raw": 0.70,
            "p_like_prior_corrected": 0.75,
        },
    ]


def make_assignments():
    """构造对应 assignments。"""

    return [
        {
            "sample_id": "a",
            "outer_fold": 1,
            "label": 1,
        },
        {
            "sample_id": "b",
            "outer_fold": 1,
            "label": 0,
        },
        {
            "sample_id": "c",
            "outer_fold": 2,
            "label": 1,
        },
        {
            "sample_id": "d",
            "outer_fold": 2,
            "label": 0,
        },
    ]


def test_enrich_and_coverage():
    """可信度字段与覆盖校验应正确。"""

    enriched = enrich_oof_predictions(
        make_predictions(),
        threshold=0.5,
    )

    by_id = {
        row["sample_id"]: row
        for row in enriched
    }

    assert by_id["a"]["label_confidence"] == pytest.approx(0.9)
    assert by_id["b"]["label_confidence"] == pytest.approx(0.8)
    assert by_id["c"]["label_agreement"] is False
    assert by_id["d"]["label_agreement"] is False

    coverage = validate_oof_coverage(
        enriched,
        make_assignments(),
    )

    assert coverage["predicted_samples"] == 4
    assert coverage["fold_counts"] == {
        "1": 2,
        "2": 2,
    }


def test_metrics_and_confidence_report():
    """指标和冲突排序应可计算。"""

    enriched = enrich_oof_predictions(
        make_predictions(),
        threshold=0.5,
    )

    metrics = compute_oof_metrics(
        enriched,
        threshold=0.5,
        ece_bins=5,
    )
    report = build_confidence_report(
        enriched,
        top_k=2,
    )

    assert metrics["num_samples"] == 4
    assert metrics["roc_auc"] is not None
    assert report["num_samples"] == 4
    assert len(report["top_conflicts"]) == 2
    assert (
        report["top_conflicts"][0]["sample_id"]
        == "d"
    )


def test_duplicate_sample_id_rejected():
    """重复 sample_id 必须报错。"""

    predictions = make_predictions()
    predictions.append(dict(predictions[0]))

    with pytest.raises(
        RuntimeError,
        match="sample_id 重复",
    ):
        enrich_oof_predictions(
            predictions,
            threshold=0.5,
        )
