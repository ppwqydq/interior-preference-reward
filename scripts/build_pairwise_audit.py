#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""生成 Pairwise 评估视觉审计三联图。

每张审计图包含：

    Reference | GOOD | BAD

并在顶部标注：

- pair_id；
- GOOD reward；
- BAD reward；
- GOOD-BAD margin；
- 当前排序是否正确。

该脚本只读取已经生成的评估结果，不加载模型，
也不修改训练、推理或评估逻辑。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="生成 Pairwise 评估三联视觉审计图。"
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="pairwise_predictions.jsonl 路径。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="视觉审计结果目录。",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="每个类别最多输出的样本数。",
    )
    parser.add_argument(
        "--panel_width",
        type=int,
        default=512,
        help="单张图片面板宽度。",
    )
    parser.add_argument(
        "--panel_height",
        type=int,
        default=384,
        help="单张图片面板高度。",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """解析项目相对路径或绝对路径。"""

    path = path.expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT / path
    ).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件。"""

    if not path.is_file():
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )

    rows: List[Dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON 解析失败：{path}:{line_number}"
                ) from exc

            if not isinstance(row, dict):
                raise TypeError(
                    f"第 {line_number} 行不是 JSON 对象"
                )

            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"文件中没有评估结果：{path}"
        )

    return rows


def load_font(size: int) -> ImageFont.ImageFont:
    """加载常见字体，缺失时退回 Pillow 默认字体。"""

    candidates = [
        Path(
            "/usr/share/fonts/truetype/"
            "dejavu/DejaVuSans.ttf"
        ),
        Path(
            "/usr/share/fonts/truetype/"
            "liberation2/LiberationSans-Regular.ttf"
        ),
    ]

    for font_path in candidates:
        if font_path.is_file():
            return ImageFont.truetype(
                str(font_path),
                size=size,
            )

    return ImageFont.load_default()


def fit_image(
    image_path: Path,
    width: int,
    height: int,
) -> Image.Image:
    """等比例缩放图片并放置到白色面板中央。"""

    if not image_path.is_file():
        raise FileNotFoundError(
            f"图片不存在：{image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        contained = ImageOps.contain(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
        )

    panel = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    offset_x = (
        width - contained.width
    ) // 2
    offset_y = (
        height - contained.height
    ) // 2

    panel.paste(
        contained,
        (offset_x, offset_y),
    )

    return panel


def status_text(row: Dict[str, Any]) -> str:
    """返回样本排序状态。"""

    if bool(row.get("is_correct")):
        return "CORRECT"

    if bool(row.get("is_tie")):
        return "TIE"

    return "WRONG"


def create_triptych(
    row: Dict[str, Any],
    output_path: Path,
    panel_width: int,
    panel_height: int,
) -> None:
    """生成单个 R/GOOD/BAD 三联图。"""

    reference_path = Path(
        row["reference_image_path"]
    )
    positive_path = Path(
        row["positive_image_path"]
    )
    negative_path = Path(
        row["negative_image_path"]
    )

    label_height = 44
    header_height = 92
    footer_height = 22
    gap = 8

    total_width = (
        panel_width * 3
        + gap * 2
    )
    total_height = (
        header_height
        + label_height
        + panel_height
        + footer_height
    )

    canvas = Image.new(
        "RGB",
        (total_width, total_height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    title_font = load_font(24)
    label_font = load_font(22)
    small_font = load_font(17)

    pair_id = str(row["pair_id"])
    good_score = float(
        row["positive_reward_score"]
    )
    bad_score = float(
        row["negative_reward_score"]
    )
    margin = float(
        row["pairwise_margin"]
    )
    status = status_text(row)

    title = (
        f"Pair {pair_id} | {status} | "
        f"GOOD={good_score:.6f} | "
        f"BAD={bad_score:.6f}"
    )
    subtitle = (
        f"Margin (GOOD - BAD) = {margin:.6f}"
    )

    draw.text(
        (16, 12),
        title,
        fill="black",
        font=title_font,
    )
    draw.text(
        (16, 50),
        subtitle,
        fill="black",
        font=small_font,
    )

    panels = [
        (
            "REFERENCE",
            reference_path,
        ),
        (
            "GOOD",
            positive_path,
        ),
        (
            "BAD",
            negative_path,
        ),
    ]

    image_y = (
        header_height
        + label_height
    )

    for index, (
        label,
        image_path,
    ) in enumerate(panels):
        x = index * (
            panel_width + gap
        )

        label_box = draw.textbbox(
            (0, 0),
            label,
            font=label_font,
        )

        label_width = (
            label_box[2] - label_box[0]
        )

        label_x = (
            x
            + (
                panel_width
                - label_width
            )
            // 2
        )

        draw.text(
            (
                label_x,
                header_height + 8,
            ),
            label,
            fill="black",
            font=label_font,
        )

        panel = fit_image(
            image_path=image_path,
            width=panel_width,
            height=panel_height,
        )

        canvas.paste(
            panel,
            (x, image_y),
        )

        draw.rectangle(
            [
                x,
                image_y,
                x + panel_width - 1,
                image_y + panel_height - 1,
            ],
            outline="black",
            width=1,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        format="PNG",
        optimize=True,
    )


def write_jsonl(
    rows: Iterable[Dict[str, Any]],
    output_path: Path,
) -> None:
    """保存分类后的审计索引。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def export_category(
    category_name: str,
    rows: List[Dict[str, Any]],
    output_dir: Path,
    panel_width: int,
    panel_height: int,
) -> None:
    """输出一个审计类别。"""

    category_dir = (
        output_dir / category_name
    )
    category_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_rows: List[
        Dict[str, Any]
    ] = []

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        pair_id = str(
            row["pair_id"]
        )

        image_name = (
            f"{rank:02d}_"
            f"pair_{pair_id}_"
            f"margin_"
            f"{float(row['pairwise_margin']):+.6f}"
            ".png"
        )

        image_path = (
            category_dir / image_name
        )

        create_triptych(
            row=row,
            output_path=image_path,
            panel_width=panel_width,
            panel_height=panel_height,
        )

        index_rows.append(
            {
                "rank": rank,
                "pair_id": pair_id,
                "status": status_text(row),
                "positive_reward_score": float(
                    row[
                        "positive_reward_score"
                    ]
                ),
                "negative_reward_score": float(
                    row[
                        "negative_reward_score"
                    ]
                ),
                "pairwise_margin": float(
                    row["pairwise_margin"]
                ),
                "audit_image_path": str(
                    image_path
                ),
                "reference_image_path": row[
                    "reference_image_path"
                ],
                "positive_image_path": row[
                    "positive_image_path"
                ],
                "negative_image_path": row[
                    "negative_image_path"
                ],
                "metadata": row.get(
                    "metadata",
                    {},
                ),
            }
        )

    write_jsonl(
        index_rows,
        category_dir / "index.jsonl",
    )


def main() -> None:
    """生成三类视觉审计结果。"""

    args = parse_args()

    if args.top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0"
        )

    if (
        args.panel_width <= 0
        or args.panel_height <= 0
    ):
        raise ValueError(
            "面板宽高必须大于 0"
        )

    predictions_path = resolve_path(
        args.predictions
    )
    output_dir = resolve_path(
        args.output_dir
    )

    rows = read_jsonl(
        predictions_path
    )

    worst_errors = sorted(
        [
            row
            for row in rows
            if bool(row.get("is_incorrect"))
        ],
        key=lambda row: float(
            row["pairwise_margin"]
        ),
    )[:args.top_k]

    difficult = sorted(
        rows,
        key=lambda row: abs(
            float(
                row["pairwise_margin"]
            )
        ),
    )[:args.top_k]

    strongest_correct = sorted(
        [
            row
            for row in rows
            if bool(row.get("is_correct"))
        ],
        key=lambda row: float(
            row["pairwise_margin"]
        ),
        reverse=True,
    )[:args.top_k]

    export_category(
        category_name="worst_errors",
        rows=worst_errors,
        output_dir=output_dir,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
    )

    export_category(
        category_name="difficult",
        rows=difficult,
        output_dir=output_dir,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
    )

    export_category(
        category_name="strongest_correct",
        rows=strongest_correct,
        output_dir=output_dir,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
    )

    summary = {
        "predictions_path": str(
            predictions_path
        ),
        "output_dir": str(
            output_dir
        ),
        "top_k": int(
            args.top_k
        ),
        "worst_error_pair_ids": [
            str(row["pair_id"])
            for row in worst_errors
        ],
        "difficult_pair_ids": [
            str(row["pair_id"])
            for row in difficult
        ],
        "strongest_correct_pair_ids": [
            str(row["pair_id"])
            for row in strongest_correct
        ],
    }

    (
        output_dir
        / "audit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        "最严重错误：",
        summary[
            "worst_error_pair_ids"
        ],
    )
    print(
        "最困难样本：",
        summary[
            "difficult_pair_ids"
        ],
    )
    print(
        "最强正确样本：",
        summary[
            "strongest_correct_pair_ids"
        ],
    )
    print(
        "视觉审计目录：",
        output_dir,
    )
    print(
        "PAIRWISE_AUDIT_BUILD_PASSED"
    )


if __name__ == "__main__":
    main()
