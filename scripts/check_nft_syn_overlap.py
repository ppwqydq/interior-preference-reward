#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(
    "/root/qwen_pref_reward"
).resolve()

NEW_MANIFEST = (
    PROJECT_ROOT
    / "data/external/dataset_nft_syn/"
      "pairwise_test.jsonl"
)

OLD_MANIFESTS = [
    PROJECT_ROOT
    / "data/splits/layout100_curated/"
      "train_pairwise.jsonl",
    PROJECT_ROOT
    / "data/splits/layout100_curated/"
      "val_pairwise.jsonl",
]

FIELDS = (
    "reference_image_path",
    "positive_image_path",
    "negative_image_path",
)


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(
            1024 * 1024
        ):
            digest.update(block)

    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def build_index(
    manifests: list[Path],
) -> dict[str, list[dict]]:
    index: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    cache: dict[Path, str] = {}

    for manifest in manifests:
        for row in read_rows(manifest):
            for field in FIELDS:
                path = resolve_path(
                    row[field]
                )

                if not path.is_file():
                    raise FileNotFoundError(
                        path
                    )

                if path not in cache:
                    cache[path] = sha256(path)

                index[cache[path]].append(
                    {
                        "manifest": str(
                            manifest
                        ),
                        "pair_id": str(
                            row["pair_id"]
                        ),
                        "role": field,
                        "path": str(path),
                    }
                )

    return index


new_index = build_index(
    [NEW_MANIFEST]
)
old_index = build_index(
    OLD_MANIFESTS
)

overlap_hashes = sorted(
    set(new_index)
    & set(old_index)
)

internal_duplicates = {
    digest: locations
    for digest, locations
    in new_index.items()
    if len(locations) > 1
}

report = {
    "new_pairs": len(
        read_rows(NEW_MANIFEST)
    ),
    "old_pairs": sum(
        len(read_rows(path))
        for path in OLD_MANIFESTS
    ),
    "cross_dataset_overlap_hashes": (
        len(overlap_hashes)
    ),
    "new_internal_duplicate_hashes": (
        len(internal_duplicates)
    ),
    "cross_dataset_overlaps": [
        {
            "sha256": digest,
            "new": new_index[digest],
            "old": old_index[digest],
        }
        for digest in overlap_hashes
    ],
    "new_internal_duplicates": (
        internal_duplicates
    ),
}

output = (
    NEW_MANIFEST.parent
    / "content_overlap_report.json"
)

output.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        {
            "new_pairs": report[
                "new_pairs"
            ],
            "old_pairs": report[
                "old_pairs"
            ],
            "cross_dataset_overlap_hashes": (
                report[
                    "cross_dataset_overlap_hashes"
                ]
            ),
            "new_internal_duplicate_hashes": (
                report[
                    "new_internal_duplicate_hashes"
                ]
            ),
            "report": str(output),
        },
        ensure_ascii=False,
        indent=2,
    )
)

if overlap_hashes:
    print(
        "\n发现与训练/验证数据完全相同的图片："
    )

    for digest in overlap_hashes:
        print(
            json.dumps(
                {
                    "new": new_index[digest],
                    "old": old_index[digest],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    raise SystemExit(2)

print(
    "\n未发现与原 Layout100 的"
    "图片内容哈希重叠。"
)
