#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""生成分 Fold、分类别的 OOF 人工审计集。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


from preference_reward.evaluation.oof_audit import (
    build_audit_set,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "从完整 OOF 预测中，按 Fold 和原标签等额选择"
            "最低可信度样本，生成可填写的人工审计集。"
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "qwen3_vl_8b_layout_ab_512_oof"
            / "oof_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "reports"
            / "oof_audit_80"
        ),
    )
    parser.add_argument(
        "--per_fold_per_label",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--asset_mode",
        choices=[
            "symlink",
            "copy",
            "none",
        ],
        default="symlink",
        help=(
            "symlink：创建图片软链接；"
            "copy：复制图片；"
            "none：HTML 直接引用原路径。"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """构建并打印审计报告。"""

    args = parse_args()

    summary = build_audit_set(
        predictions_path=(
            args.predictions.resolve()
        ),
        output_dir=(
            args.output_dir.resolve()
        ),
        per_fold_per_label=(
            args.per_fold_per_label
        ),
        asset_mode=args.asset_mode,
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
