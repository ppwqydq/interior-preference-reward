#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

ROLE_PATTERN = re.compile(
    r"^(?P<pair_id>.+)_(?P<role>R|good|bad|preview|meta)$",
    re.IGNORECASE,
)


def natural_key(value: str) -> tuple:
    return tuple(
        int(part) if part.isdigit()
        else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def load_meta(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return {
            "_parse_error": str(exc),
            "_meta_path": str(path),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    groups: dict[
        str,
        dict[str, list[Path]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue

        match = ROLE_PATTERN.match(path.stem)

        if match is None:
            continue

        pair_id = match.group("pair_id").strip()
        role = match.group("role").lower()

        if (
            role in {"r", "good", "bad", "preview"}
            and path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        groups[pair_id][role].append(
            path.resolve()
        )

    rows: list[dict[str, Any]] = []
    reference_only: list[str] = []
    broken: list[str] = []

    for pair_id in sorted(
        groups,
        key=natural_key,
    ):
        roles = groups[pair_id]

        required_counts = {
            role: len(roles.get(role, []))
            for role in ("r", "good", "bad")
        }

        if all(
            count == 1
            for count in required_counts.values()
        ):
            row: dict[str, Any] = {
                "pair_id": pair_id,
                "reference_image_path": str(
                    roles["r"][0]
                ),
                "positive_image_path": str(
                    roles["good"][0]
                ),
                "negative_image_path": str(
                    roles["bad"][0]
                ),
                "positive_role": "good",
                "negative_role": "bad",
                "expected_order": (
                    "positive>negative"
                ),
                "source_dataset": (
                    "dataset_nft_syn"
                ),
            }

            if len(roles.get("meta", [])) == 1:
                meta_path = roles["meta"][0]
                row["meta_path"] = str(
                    meta_path
                )
                row["source_meta"] = load_meta(
                    meta_path
                )

            if len(
                roles.get("preview", [])
            ) == 1:
                row["preview_path"] = str(
                    roles["preview"][0]
                )

            rows.append(row)
            continue

        if (
            required_counts["r"] == 1
            and required_counts["good"] == 0
            and required_counts["bad"] == 0
        ):
            reference_only.append(pair_id)
            continue

        broken.append(
            f"{pair_id}: {required_counts}"
        )

    if broken:
        raise RuntimeError(
            "仍有不完整或重复的候选组：\n"
            + "\n".join(broken)
        )

    if not rows:
        raise RuntimeError(
            "没有完整 Pair"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"完整 Pair：{len(rows)}")
    print(
        f"仅参考图资产："
        f"{len(reference_only)}"
    )
    print(f"输出：{output}")

    print("\nPair ID：")
    print(
        [
            row["pair_id"]
            for row in rows
        ]
    )


if __name__ == "__main__":
    main()
