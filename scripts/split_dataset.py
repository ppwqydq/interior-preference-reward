#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""将清洗后的样本分组划分为训练集和验证集。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from preference_reward.data.splitter import split_dataset


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "按照空房间图片分组划分训练集和验证集，"
            "避免同一空间泄漏。"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "samples.jsonl"
        ),
    )
    parser.add_argument(
        "--train_output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "train.jsonl"
        ),
    )
    parser.add_argument(
        "--val_output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "val.jsonl"
        ),
    )
    parser.add_argument(
        "--report_output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "reports"
            / "split_report.json"
        ),
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--search_trials",
        type=int,
        default=5000,
    )

    return parser.parse_args()


def main() -> None:
    """执行数据划分。"""

    args = parse_args()

    args.train_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.val_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = split_dataset(
        input_manifest=args.input.resolve(),
        train_output=args.train_output.resolve(),
        validation_output=args.val_output.resolve(),
        report_output=args.report_output.resolve(),
        validation_ratio=args.val_ratio,
        seed=args.seed,
        search_trials=args.search_trials,
    )

    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
