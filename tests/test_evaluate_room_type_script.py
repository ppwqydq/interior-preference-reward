#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts/evaluate_room_type.py"
)


def load_script_module() -> ModuleType:
    """加载独立脚本，便于测试其中的小函数。"""

    specification = (
        importlib.util.spec_from_file_location(
            "evaluate_room_type_script",
            SCRIPT_PATH,
        )
    )

    assert specification is not None
    assert specification.loader is not None

    module = importlib.util.module_from_spec(
        specification
    )
    specification.loader.exec_module(module)

    return module


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    """写入测试 JSONL。"""

    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_normalize_prediction_aliases() -> None:
    module = load_script_module()

    normalized = (
        module.normalize_prediction_record(
            {
                "sample_id": "sample-1",
                "label": 1,
                "reward_score": 1.25,
                "probability_like_raw": 0.8,
                "probability_like_prior_corrected": (
                    0.7
                ),
            },
            line_number=1,
        )
    )

    assert normalized["label"] == 1
    assert normalized["reward_score"] == 1.25
    assert normalized["p_like_raw"] == 0.8
    assert (
        normalized[
            "p_like_prior_corrected"
        ]
        == 0.7
    )


def test_missing_corrected_probability_uses_raw() -> None:
    module = load_script_module()

    normalized = (
        module.normalize_prediction_record(
            {
                "sample_id": "sample-1",
                "label": 0,
                "reward_score": -1.0,
                "p_like_raw": 0.2,
            },
            line_number=1,
        )
    )

    assert (
        normalized[
            "p_like_prior_corrected"
        ]
        == 0.2
    )


def test_default_output_path(
    tmp_path: Path,
) -> None:
    module = load_script_module()

    predictions_path = (
        tmp_path
        / "epoch_3"
        / "val_predictions.jsonl"
    )

    output_path = module.build_output_path(
        predictions_path=predictions_path,
        requested_output=None,
    )

    assert output_path == (
        predictions_path.parent
        / "room_type_metrics.json"
    )


def test_build_diagnostic_report(
    tmp_path: Path,
) -> None:
    module = load_script_module()

    train_manifest = (
        tmp_path / "train.jsonl"
    )
    val_manifest = tmp_path / "val.jsonl"
    predictions_path = (
        tmp_path / "val_predictions.jsonl"
    )

    write_jsonl(
        train_manifest,
        [
            {
                "sample_id": "t1",
                "empty_room_image": "t1-empty.png",
                "generated_furniture_image": (
                    "t1-generated.png"
                ),
                "label": 1,
                "room_type": "Living Room",
            },
            {
                "sample_id": "t2",
                "empty_room_image": "t2-empty.png",
                "generated_furniture_image": (
                    "t2-generated.png"
                ),
                "label": 0,
                "room_type": "Living Room",
            },
            {
                "sample_id": "t3",
                "empty_room_image": "t3-empty.png",
                "generated_furniture_image": (
                    "t3-generated.png"
                ),
                "label": 1,
                "room_type": "Bathroom",
            },
            {
                "sample_id": "t4",
                "empty_room_image": "t4-empty.png",
                "generated_furniture_image": (
                    "t4-generated.png"
                ),
                "label": 0,
                "room_type": "Bathroom",
            },
        ],
    )

    write_jsonl(
        val_manifest,
        [
            {
                "sample_id": "v1",
                "empty_room_image": "v1-empty.png",
                "generated_furniture_image": (
                    "v1-generated.png"
                ),
                "label": 1,
                "room_type": "Living Room",
            },
            {
                "sample_id": "v2",
                "empty_room_image": "v2-empty.png",
                "generated_furniture_image": (
                    "v2-generated.png"
                ),
                "label": 0,
                "room_type": "Living Room",
            },
            {
                "sample_id": "v3",
                "empty_room_image": "v3-empty.png",
                "generated_furniture_image": (
                    "v3-generated.png"
                ),
                "label": 1,
                "room_type": "Bathroom",
            },
            {
                "sample_id": "v4",
                "empty_room_image": "v4-empty.png",
                "generated_furniture_image": (
                    "v4-generated.png"
                ),
                "label": 0,
                "room_type": "Bathroom",
            },
        ],
    )

    write_jsonl(
        predictions_path,
        [
            {
                "sample_id": "v1",
                "label": 1,
                "reward_score": 2.0,
                "p_like_raw": 0.9,
                "p_like_prior_corrected": 0.9,
            },
            {
                "sample_id": "v2",
                "label": 0,
                "reward_score": -2.0,
                "p_like_raw": 0.1,
                "p_like_prior_corrected": 0.1,
            },
            {
                "sample_id": "v3",
                "label": 1,
                "reward_score": 1.0,
                "p_like_raw": 0.8,
                "p_like_prior_corrected": 0.8,
            },
            {
                "sample_id": "v4",
                "label": 0,
                "reward_score": -1.0,
                "p_like_raw": 0.2,
                "p_like_prior_corrected": 0.2,
            },
        ],
    )

    report = module.build_diagnostic_report(
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        predictions_path=predictions_path,
        threshold=0.5,
        ece_bins=5,
    )

    assert (
        report["model_overall"]["roc_auc"]
        == 1.0
    )

    assert set(
        report["model_by_room_type"]
    ) == {
        "Bathroom",
        "Living Room",
    }

    assert (
        report[
            "model_by_room_type_summary"
        ]["macro_roc_auc"]
        == 1.0
    )
