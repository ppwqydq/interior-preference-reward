#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from preference_reward.data.room_type_manifest import (
    make_csv_key,
    make_manifest_key,
    normalize_text,
    sha256_text,
)


def test_normalize_text() -> None:
    assert normalize_text(None) == ""
    assert normalize_text(" Bedroom ") == "Bedroom"


def test_sha256_text_is_stable() -> None:
    assert sha256_text("example") == sha256_text(
        "example"
    )
    assert len(sha256_text("example")) == 64


def test_csv_and_manifest_keys_match() -> None:
    empty_url = (
        "https://example.com/empty.png"
    )
    generated_url = (
        "https://example.com/generated.png"
    )

    csv_key = make_csv_key(
        empty_room_url=empty_url,
        generated_furniture_url=generated_url,
        label=1,
    )

    manifest_key = make_manifest_key(
        {
            "empty_room_image": (
                "data/images/empty_room/"
                f"{sha256_text(empty_url)}.png"
            ),
            "generated_furniture_image": (
                "data/images/generated_furniture/"
                f"{sha256_text(generated_url)}.png"
            ),
            "label": 1,
        }
    )

    assert csv_key == manifest_key
