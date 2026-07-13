#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""生成固定空间布局外部测试集清单。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from preference_reward.data.layout_test_builder import (
    build_layout_test,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "根据 R/GOOD/BAD 图片构建空间布局外部测试集。"
        )
    )

    parser.add_argument(
        "--source_dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "external"
            / "layout_test"
            / "raw"
        ),
    )
    parser.add_argument(
        "--training_manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "samples.jsonl"
        ),
    )
    parser.add_argument(
        "--pointwise_output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "external"
            / "layout_test"
            / "pointwise_test.jsonl"
        ),
    )
    parser.add_argument(
        "--pairwise_output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "external"
            / "layout_test"
            / "pairwise_test.jsonl"
        ),
    )
    parser.add_argument(
        "--report_output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "external"
            / "layout_test"
            / "test_manifest_report.json"
        ),
    )
    parser.add_argument(
        "--allow_training_overlap",
        action="store_true",
        help="允许测试图片与训练数据内容重复，不建议使用。",
    )

    return parser.parse_args()


def main() -> None:
    """构建并输出测试清单。"""

    args = parse_args()

    report = build_layout_test(
        project_root=PROJECT_ROOT,
        source_dir=args.source_dir.resolve(),
        training_manifest=args.training_manifest.resolve(),
        pointwise_output=args.pointwise_output.resolve(),
        pairwise_output=args.pairwise_output.resolve(),
        report_output=args.report_output.resolve(),
        allow_training_overlap=args.allow_training_overlap,
    )

    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
