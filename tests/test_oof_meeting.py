#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OOF 会议图集基础测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from preference_reward.evaluation.oof_meeting import (
    build_meeting_gallery,
    read_audit_csv,
)


def _write_image(path: Path, value: int) -> None:
    """写入测试图片。"""

    image = Image.new(
        "RGB",
        (320, 240),
        (value, value, value),
    )
    image.save(path)


def _write_csv(
    path: Path,
    image_dir: Path,
) -> None:
    """写入四类测试审计 CSV。"""

    fields = [
        "audit_rank",
        "fold",
        "label",
        "sample_id",
        "label_confidence",
        "reward_score",
        "p_like_prior_corrected",
        "empty_room_image",
        "generated_furniture_image",
        "review_decision",
        "reviewer_notes",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()

        for index, decision in enumerate(
            ("A", "B", "C", "D"),
            start=1,
        ):
            writer.writerow(
                {
                    "audit_rank": index,
                    "fold": index,
                    "label": index % 2,
                    "sample_id": (
                        decision.lower() * 64
                    )[:64],
                    "label_confidence": (
                        0.1 * index
                    ),
                    "reward_score": (
                        float(index)
                    ),
                    "p_like_prior_corrected": (
                        0.8
                        if index % 2
                        else 0.2
                    ),
                    "empty_room_image": str(
                        image_dir / "empty.png"
                    ),
                    "generated_furniture_image": str(
                        image_dir / "generated.png"
                    ),
                    "review_decision": decision,
                    "reviewer_notes": (
                        f"notes-{decision}"
                    ),
                }
            )


def test_read_and_build_gallery(
    tmp_path: Path,
):
    """四类图集应完整生成。"""

    _write_image(
        tmp_path / "empty.png",
        220,
    )
    _write_image(
        tmp_path / "generated.png",
        180,
    )

    csv_path = (
        tmp_path / "audit.csv"
    )
    _write_csv(
        csv_path,
        tmp_path,
    )

    rows = read_audit_csv(
        csv_path
    )

    assert len(rows) == 4

    output_dir = (
        tmp_path / "meeting"
    )
    summary = build_meeting_gallery(
        audit_csv=csv_path,
        output_dir=output_dir,
        items_per_sheet=1,
    )

    assert summary["total_samples"] == 4
    assert summary["category_counts"] == {
        "A": 1,
        "B": 1,
        "C": 1,
        "D": 1,
    }
    assert (
        output_dir / "index.html"
    ).is_file()
    assert (
        output_dir
        / "contact_sheets"
        / "00_overview.jpg"
    ).is_file()

    for directory in (
        "A_label_error",
        "B_subjective_preference",
        "C_model_error",
        "D_uncertain_or_abnormal",
    ):
        assert len(
            list(
                (
                    output_dir
                    / directory
                ).glob("*.jpg")
            )
        ) == 1


def test_invalid_decision_rejected(
    tmp_path: Path,
):
    """无效分类必须报错。"""

    _write_image(
        tmp_path / "empty.png",
        220,
    )
    _write_image(
        tmp_path / "generated.png",
        180,
    )

    csv_path = (
        tmp_path / "audit.csv"
    )
    _write_csv(
        csv_path,
        tmp_path,
    )

    text = csv_path.read_text(
        encoding="utf-8-sig"
    )
    csv_path.write_text(
        text.replace(",D,", ",X,"),
        encoding="utf-8-sig",
    )

    with pytest.raises(ValueError):
        read_audit_csv(csv_path)
