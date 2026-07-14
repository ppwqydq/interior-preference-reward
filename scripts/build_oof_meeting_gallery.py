#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""生成分类三联图和会议联系表。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


from preference_reward.evaluation.oof_meeting import (
    build_meeting_gallery,
)


def parse_args() -> argparse.Namespace:
    """解析参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "根据填写完成的 audit_template.csv，"
            "按 A/B/C/D 生成三联图、分类图集和会议联系表。"
        )
    )
    parser.add_argument(
        "--audit_csv",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "reports"
            / "oof_audit_80"
            / "audit_template.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "reports"
            / "oof_audit_meeting"
        ),
    )
    parser.add_argument(
        "--items_per_sheet",
        type=int,
        default=2,
        help="每张 16:9 联系表放几组三联图。",
    )

    return parser.parse_args()


def main() -> None:
    """执行会议图集生成。"""

    args = parse_args()

    summary = build_meeting_gallery(
        audit_csv=args.audit_csv.resolve(),
        output_dir=(
            args.output_dir.resolve()
        ),
        items_per_sheet=(
            args.items_per_sheet
        ),
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
