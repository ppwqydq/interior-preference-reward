#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""构建按空房间分组的 OOF 外折和折内训练/验证清单。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from preference_reward.data.oof_splitter import (
    build_oof_splits,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "仅使用现有训练集，按 empty_room_image "
            "分组构建 OOF 外折与折内验证集。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "train.jsonl"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "oof_4fold"
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--inner_val_ratio",
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
    """执行 OOF 划分。"""

    args = parse_args()

    report = build_oof_splits(
        input_manifest=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        folds=args.folds,
        inner_validation_ratio=(
            args.inner_val_ratio
        ),
        seed=args.seed,
        search_trials=args.search_trials,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
