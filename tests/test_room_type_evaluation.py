#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pytest

from preference_reward.data.manifest import (
    PreferenceSample,
)
from preference_reward.evaluation.room_type import (
    evaluate_by_room_type,
    normalize_room_type,
)


def make_sample(
    sample_id: str,
    label: int,
    room_type: str,
) -> PreferenceSample:
    return PreferenceSample(
        sample_id=sample_id,
        empty_room_image=Path(
            f"/tmp/{sample_id}_empty.png"
        ),
        generated_furniture_image=Path(
            f"/tmp/{sample_id}_generated.png"
        ),
        label=label,
        room_type=room_type,
    )


def make_prediction(
    sample: PreferenceSample,
    reward_score: float,
    probability: float,
) -> dict:
    return {
        "sample_id": sample.sample_id,
        "label": sample.label,
        "reward_score": reward_score,
        "p_like_raw": probability,
        "p_like_prior_corrected": (
            probability
        ),
        "prediction_raw": int(
            probability >= 0.5
        ),
        "prediction": int(
            probability >= 0.5
        ),
    }


def test_normalize_unknown_room_type() -> None:
    assert normalize_room_type(
        "Unknown"
    ) == "Unknown"
    assert normalize_room_type(
        ""
    ) == "Unknown"
    assert normalize_room_type(
        " Bedroom "
    ) == "Bedroom"


def test_room_type_evaluation() -> None:
    train_samples = [
        make_sample("t1", 1, "Living Room"),
        make_sample("t2", 1, "Living Room"),
        make_sample("t3", 0, "Living Room"),
        make_sample("t4", 0, "Bathroom"),
        make_sample("t5", 0, "Bathroom"),
        make_sample("t6", 1, "Bathroom"),
    ]

    val_samples = [
        make_sample("v1", 1, "Living Room"),
        make_sample("v2", 0, "Living Room"),
        make_sample("v3", 1, "Bathroom"),
        make_sample("v4", 0, "Bathroom"),
    ]

    predictions = [
        make_prediction(
            val_samples[0],
            reward_score=2.0,
            probability=0.90,
        ),
        make_prediction(
            val_samples[1],
            reward_score=-2.0,
            probability=0.10,
        ),
        make_prediction(
            val_samples[2],
            reward_score=1.0,
            probability=0.75,
        ),
        make_prediction(
            val_samples[3],
            reward_score=-1.0,
            probability=0.25,
        ),
    ]

    report = evaluate_by_room_type(
        train_samples=train_samples,
        val_samples=val_samples,
        predictions=predictions,
        threshold=0.5,
        ece_bins=5,
    )

    assert set(
        report["model_by_room_type"]
    ) == {
        "Bathroom",
        "Living Room",
    }

    assert (
        report["model_by_room_type"]
        ["Living Room"]["roc_auc"]
        == pytest.approx(1.0)
    )
    assert (
        report["model_by_room_type"]
        ["Bathroom"]["roc_auc"]
        == pytest.approx(1.0)
    )

    summary = report[
        "model_by_room_type_summary"
    ]

    assert summary["valid_room_types"] == 2
    assert summary[
        "macro_roc_auc"
    ] == pytest.approx(1.0)
    assert summary[
        "weighted_roc_auc"
    ] == pytest.approx(1.0)

    baseline = report[
        "room_type_only_baseline"
    ]

    assert baseline["num_samples"] == 4
    assert baseline["roc_auc"] is not None


def test_alignment_error() -> None:
    train_samples = [
        make_sample("t1", 1, "Bedroom"),
        make_sample("t2", 0, "Bedroom"),
    ]

    val_samples = [
        make_sample("v1", 1, "Bedroom"),
    ]

    predictions = [
        {
            "sample_id": "wrong",
            "label": 1,
            "reward_score": 1.0,
            "p_like_raw": 0.8,
            "p_like_prior_corrected": 0.8,
        }
    ]

    with pytest.raises(
        ValueError,
        match="sample_id",
    ):
        evaluate_by_room_type(
            train_samples=train_samples,
            val_samples=val_samples,
            predictions=predictions,
            threshold=0.5,
            ece_bins=5,
        )
