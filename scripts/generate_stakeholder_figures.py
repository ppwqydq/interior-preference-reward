#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(
    "/root/qwen_pref_reward"
).resolve()

ORIGINAL_MANIFEST = (
    PROJECT_ROOT
    / "data/external/dataset_nft_syn/"
      "pairwise_test.jsonl"
)

FILTERED_MANIFEST = (
    PROJECT_ROOT
    / "data/external/dataset_nft_syn/"
      "pairwise_test_filtered.jsonl"
)

EVAL_ROOT = Path(
    "/root/autodl-tmp/qwen_pref_reward_outputs/"
    "dataset_nft_syn_filtered_evaluation"
)

P1_PREDICTIONS = (
    EVAL_ROOT
    / "p1_epoch23/pairwise_predictions.jsonl"
)

P2_PREDICTIONS = (
    EVAL_ROOT
    / "p2_epoch23/pairwise_predictions.jsonl"
)

P1_METRICS = (
    EVAL_ROOT
    / "p1_epoch23/metrics.json"
)

P2_METRICS = (
    EVAL_ROOT
    / "p2_epoch23/metrics.json"
)

OUTPUT_DIR = Path(
    "/root/autodl-tmp/qwen_pref_reward_outputs/"
    "stakeholder_figures"
).resolve()


BG = "#F7F8FA"
CARD = "#FFFFFF"
TEXT = "#172033"
MUTED = "#586174"
BORDER = "#D8DDE6"
GOOD = "#2E8B57"
BAD = "#C74747"
ACCENT = "#315E9E"
ACCENT_2 = "#8155A3"
WARNING = "#C98118"
LIGHT_BLUE = "#EAF1FB"
LIGHT_GREEN = "#EAF7EF"
LIGHT_RED = "#FCEDED"
LIGHT_YELLOW = "#FFF5DF"


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def resolve_image_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def find_font_file() -> str | None:
    families = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "DejaVu Sans",
    ]

    for family in families:
        try:
            result = subprocess.run(
                [
                    "fc-match",
                    "-f",
                    "%{file}",
                    family,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            value = result.stdout.strip()

            if value and Path(value).is_file():
                return value

        except Exception:
            continue

    known_paths = [
        "/usr/share/fonts/opentype/noto/"
        "NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/"
        "wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans.ttf",
    ]

    for value in known_paths:
        if Path(value).is_file():
            return value

    return None


FONT_FILE = find_font_file()


def font(size: int, bold: bool = False):
    if FONT_FILE is None:
        return ImageFont.load_default()

    if bold:
        candidates = [
            FONT_FILE.replace(
                "Regular",
                "Bold",
            ),
            FONT_FILE.replace(
                "regular",
                "bold",
            ),
        ]

        for candidate in candidates:
            if Path(candidate).is_file():
                return ImageFont.truetype(
                    candidate,
                    size=size,
                )

    return ImageFont.truetype(
        FONT_FILE,
        size=size,
    )


TITLE_FONT = font(54, bold=True)
SUBTITLE_FONT = font(31, bold=True)
BODY_FONT = font(27)
SMALL_FONT = font(21)
TINY_FONT = font(17)


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    fill: str = TEXT,
    selected_font=None,
    anchor: str | None = None,
):
    draw.text(
        xy,
        value,
        fill=fill,
        font=selected_font or BODY_FONT,
        anchor=anchor,
    )


def wrap_text_by_pixels(
    draw: ImageDraw.ImageDraw,
    value: str,
    selected_font,
    max_width: int,
) -> list[str]:
    result: list[str] = []

    for paragraph in value.splitlines():
        if not paragraph:
            result.append("")
            continue

        current = ""

        for char in paragraph:
            candidate = current + char

            box = draw.textbbox(
                (0, 0),
                candidate,
                font=selected_font,
            )

            width = box[2] - box[0]

            if current and width > max_width:
                result.append(current)
                current = char
            else:
                current = candidate

        if current:
            result.append(current)

    return result


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    max_width: int,
    selected_font=None,
    fill: str = TEXT,
    line_spacing: int = 9,
) -> int:
    selected_font = selected_font or BODY_FONT

    lines = wrap_text_by_pixels(
        draw,
        value,
        selected_font,
        max_width,
    )

    x, y = xy
    line_height = (
        draw.textbbox(
            (0, 0),
            "Ag国",
            font=selected_font,
        )[3]
        + line_spacing
    )

    for line in lines:
        draw.text(
            (x, y),
            line,
            fill=fill,
            font=selected_font,
        )
        y += line_height

    return y


def rounded_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = CARD,
    outline: str = BORDER,
    radius: int = 24,
    width: int = 2,
):
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def load_fitted_image(
    path: Path,
    size: tuple[int, int],
    *,
    background: str = "#FFFFFF",
) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")

    image.thumbnail(
        size,
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        size,
        background,
    )

    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2

    canvas.paste(
        image,
        (x, y),
    )

    return canvas


def add_image_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    image_path: Path,
    label: str,
    label_fill: str,
):
    rounded_card(
        draw,
        (
            x,
            y,
            x + width,
            y + height,
        ),
    )

    image = load_fitted_image(
        image_path,
        (
            width - 20,
            height - 75,
        ),
    )

    canvas.paste(
        image,
        (
            x + 10,
            y + 55,
        ),
    )

    draw_text(
        draw,
        (
            x + width // 2,
            y + 28,
        ),
        label,
        fill=label_fill,
        selected_font=SUBTITLE_FONT,
        anchor="mm",
    )


def create_triptych(
    *,
    row: dict[str, Any],
    output_path: Path,
    title: str,
    subtitle: str,
    footer: str,
    extra_header: str | None = None,
):
    width = 1800
    height = 850

    canvas = Image.new(
        "RGB",
        (width, height),
        BG,
    )
    draw = ImageDraw.Draw(canvas)

    draw_text(
        draw,
        (70, 45),
        title,
        selected_font=TITLE_FONT,
    )

    draw_wrapped(
        draw,
        (72, 115),
        subtitle,
        max_width=1640,
        selected_font=BODY_FONT,
        fill=MUTED,
    )

    if extra_header:
        draw_text(
            draw,
            (1725, 55),
            extra_header,
            selected_font=SMALL_FONT,
            fill=ACCENT,
            anchor="ra",
        )

    paths = [
        resolve_image_path(
            row["reference_image_path"]
        ),
        resolve_image_path(
            row["positive_image_path"]
        ),
        resolve_image_path(
            row["negative_image_path"]
        ),
    ]

    labels = [
        ("R：空房间", ACCENT),
        ("GOOD：较好布局", GOOD),
        ("BAD：较差布局", BAD),
    ]

    panel_width = 530
    panel_height = 500
    gap = 35
    start_x = 70
    panel_y = 205

    for index, (
        path,
        (label, color),
    ) in enumerate(
        zip(paths, labels)
    ):
        add_image_panel(
            canvas,
            draw,
            x=start_x
            + index * (
                panel_width + gap
            ),
            y=panel_y,
            width=panel_width,
            height=panel_height,
            image_path=path,
            label=label,
            label_fill=color,
        )

    rounded_card(
        draw,
        (
            70,
            735,
            1730,
            810,
        ),
        fill=LIGHT_BLUE,
        outline=LIGHT_BLUE,
        radius=18,
    )

    draw_wrapped(
        draw,
        (95, 752),
        footer,
        max_width=1600,
        selected_font=SMALL_FONT,
        fill=TEXT,
    )

    canvas.save(
        output_path,
        quality=96,
    )


def generate_01_task_definition(
    filtered_rows: list[dict[str, Any]],
):
    preferred = next(
        (
            row
            for row in filtered_rows
            if str(row["pair_id"])
            == "A_1"
        ),
        filtered_rows[0],
    )

    create_triptych(
        row=preferred,
        output_path=(
            OUTPUT_DIR
            / "01_任务定义_R_GOOD_BAD.png"
        ),
        title="任务定义：比较同一个空房间下的两个布局",
        subtitle=(
            "模型不需要给布局判定统一的绝对分数，"
            "只需要在房间与视角保持一致时，"
            "让更合理的布局获得更高 Reward。"
        ),
        footer=(
            "有效数据要求：R 为空房间；GOOD 和 BAD "
            "来自同一个房间、同一个视角；主要差异只能是家具布局。"
        ),
        extra_header=(
            f"示例 Pair：{preferred['pair_id']}"
        ),
    )


def generate_02_noise_explanation():
    width = 1800
    height = 920

    canvas = Image.new(
        "RGB",
        (width, height),
        BG,
    )
    draw = ImageDraw.Draw(canvas)

    draw_text(
        draw,
        (70, 55),
        "为什么早期用户点赞/点踩数据不够稳定",
        selected_font=TITLE_FONT,
    )

    draw_wrapped(
        draw,
        (72, 125),
        (
            "用户反馈中混合了布局质量、个人审美、误操作"
            "以及图片任务异常。它并不完全等同于“布局是否合理”。"
        ),
        max_width=1650,
        selected_font=BODY_FONT,
        fill=MUTED,
    )

    cards = [
        (
            "A",
            "可能误点或反馈错位",
            (
                "图片存在明显问题，但用户点了喜欢；"
                "或者图片基本正常，却被点踩。"
            ),
            LIGHT_RED,
            BAD,
        ),
        (
            "B",
            "个人审美和偏好",
            (
                "布局没有严重功能问题，但用户可能不喜欢"
                "家具风格、密度、颜色或摆放习惯。"
            ),
            LIGHT_YELLOW,
            WARNING,
        ),
        (
            "C",
            "模型自身判断错误",
            (
                "用户标签和布局问题合理，"
                "但模型没有识别碰撞、通道或空间关系。"
            ),
            LIGHT_BLUE,
            ACCENT,
        ),
        (
            "D",
            "样本或任务异常",
            (
                "图片、原始任务和反馈之间的关系不清楚，"
                "无法确认用户实际评价的内容。"
            ),
            "#F1ECF7",
            ACCENT_2,
        ),
    ]

    card_width = 790
    card_height = 270
    positions = [
        (70, 235),
        (940, 235),
        (70, 550),
        (940, 550),
    ]

    for (
        letter,
        heading,
        body,
        fill,
        color,
    ), (x, y) in zip(
        cards,
        positions,
    ):
        rounded_card(
            draw,
            (
                x,
                y,
                x + card_width,
                y + card_height,
            ),
            fill=fill,
            outline=fill,
        )

        draw.ellipse(
            (
                x + 35,
                y + 38,
                x + 115,
                y + 118,
            ),
            fill=color,
        )

        draw_text(
            draw,
            (
                x + 75,
                y + 78,
            ),
            letter,
            fill="#FFFFFF",
            selected_font=SUBTITLE_FONT,
            anchor="mm",
        )

        draw_text(
            draw,
            (
                x + 145,
                y + 43,
            ),
            heading,
            selected_font=SUBTITLE_FONT,
            fill=color,
        )

        draw_wrapped(
            draw,
            (
                x + 145,
                y + 105,
            ),
            body,
            max_width=600,
            selected_font=BODY_FONT,
            fill=TEXT,
        )

    draw_text(
        draw,
        (70, 865),
        (
            "说明：本图是对早期人工审计结论的概括示意，"
            "不代替真实案例截图。"
        ),
        selected_font=SMALL_FONT,
        fill=MUTED,
    )

    canvas.save(
        OUTPUT_DIR
        / "02_早期用户反馈噪声说明.png",
        quality=96,
    )


def draw_bar_chart(
    *,
    title: str,
    subtitle: str,
    bars: list[tuple[str, float, str, str]],
    output_path: Path,
    note: str,
):
    width = 1800
    height = 1000

    canvas = Image.new(
        "RGB",
        (width, height),
        BG,
    )
    draw = ImageDraw.Draw(canvas)

    draw_text(
        draw,
        (70, 50),
        title,
        selected_font=TITLE_FONT,
    )

    draw_wrapped(
        draw,
        (72, 120),
        subtitle,
        max_width=1650,
        selected_font=BODY_FONT,
        fill=MUTED,
    )

    chart_left = 180
    chart_right = 1680
    chart_top = 245
    chart_bottom = 790

    draw.line(
        (
            chart_left,
            chart_bottom,
            chart_right,
            chart_bottom,
        ),
        fill=TEXT,
        width=3,
    )

    for value in range(
        0,
        101,
        20,
    ):
        y = chart_bottom - int(
            (
                chart_bottom
                - chart_top
            )
            * value
            / 100
        )

        draw.line(
            (
                chart_left,
                y,
                chart_right,
                y,
            ),
            fill=BORDER,
            width=1,
        )

        draw_text(
            draw,
            (
                chart_left - 25,
                y,
            ),
            f"{value}%",
            selected_font=SMALL_FONT,
            fill=MUTED,
            anchor="rm",
        )

    available_width = (
        chart_right - chart_left
    )
    slot = available_width // len(bars)
    bar_width = min(
        280,
        slot - 90,
    )

    for index, (
        label,
        value,
        color,
        sample,
    ) in enumerate(bars):
        center_x = (
            chart_left
            + slot * index
            + slot // 2
        )

        bar_height = int(
            (
                chart_bottom
                - chart_top
            )
            * value
            / 100
        )

        x1 = center_x - bar_width // 2
        x2 = center_x + bar_width // 2
        y1 = chart_bottom - bar_height

        draw.rounded_rectangle(
            (
                x1,
                y1,
                x2,
                chart_bottom,
            ),
            radius=18,
            fill=color,
        )

        draw_text(
            draw,
            (
                center_x,
                y1 - 42,
            ),
            f"{value:.1f}%",
            selected_font=SUBTITLE_FONT,
            fill=color,
            anchor="mm",
        )

        draw_wrapped(
            draw,
            (
                center_x - slot // 2 + 20,
                chart_bottom + 28,
            ),
            label,
            max_width=slot - 40,
            selected_font=SMALL_FONT,
            fill=TEXT,
        )

        draw_text(
            draw,
            (
                center_x,
                chart_bottom + 112,
            ),
            sample,
            selected_font=TINY_FONT,
            fill=MUTED,
            anchor="mm",
        )

    rounded_card(
        draw,
        (
            70,
            900,
            1730,
            965,
        ),
        fill=LIGHT_YELLOW,
        outline=LIGHT_YELLOW,
        radius=16,
    )

    draw_wrapped(
        draw,
        (95, 915),
        note,
        max_width=1600,
        selected_font=SMALL_FONT,
        fill=TEXT,
    )

    canvas.save(
        output_path,
        quality=96,
    )


def generate_03_stage_results():
    draw_bar_chart(
        title="从用户反馈分类到专项布局排序",
        subtitle=(
            "早期模型学习单张图片的点赞/点踩；"
            "当前模型直接学习同一空房间下 GOOD 与 BAD 的相对顺序。"
        ),
        bars=[
            (
                "AID Feedback V2\nPointwise",
                66.0,
                ACCENT,
                "66 / 100",
            ),
            (
                "P1 Pairwise\nA/B Logit",
                100.0,
                GOOD,
                "18 / 18",
            ),
            (
                "P2 Pairwise\nScalar Head",
                94.4444,
                ACCENT_2,
                "17 / 18",
            ),
        ],
        output_path=(
            OUTPUT_DIR
            / "03_Pointwise与Pairwise阶段结果.png"
        ),
        note=(
            "注意：Pointwise 的 100 组测试集与当前 18 组"
            "任务一致性测试集不同，不能作为严格同条件模型排名；"
            "该图用于说明技术路线和阶段变化。"
        ),
    )


def format_metric(
    value: Any,
    digits: int = 4,
) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def generate_04_model_comparison(
    p1: dict[str, Any],
    p2: dict[str, Any],
):
    width = 1800
    height = 980

    canvas = Image.new(
        "RGB",
        (width, height),
        BG,
    )
    draw = ImageDraw.Draw(canvas)

    draw_text(
        draw,
        (70, 50),
        "当前两种 Pairwise Reward 方案对比",
        selected_font=TITLE_FONT,
    )

    draw_wrapped(
        draw,
        (72, 120),
        (
            "P1 复用 Qwen3-VL 原有 A/B 判断通道；"
            "P2 增加独立 Scalar Reward Head。"
        ),
        max_width=1650,
        selected_font=BODY_FONT,
        fill=MUTED,
    )

    columns = [
        ("指标", 70, 500),
        ("P1：A/B Logit", 570, 1110),
        ("P2：Scalar Head", 1130, 1730),
    ]

    header_y1 = 220
    header_y2 = 305

    for label, x1, x2 in columns:
        fill = (
            "#E9EDF3"
            if label == "指标"
            else LIGHT_BLUE
        )

        draw.rounded_rectangle(
            (
                x1,
                header_y1,
                x2,
                header_y2,
            ),
            radius=16,
            fill=fill,
        )

        draw_text(
            draw,
            (
                (x1 + x2) // 2,
                (header_y1 + header_y2) // 2,
            ),
            label,
            selected_font=SUBTITLE_FONT,
            fill=TEXT,
            anchor="mm",
        )

    rows = [
        (
            "排序正确",
            f"{p1['correct_count']} / {p1['pair_count']}",
            f"{p2['correct_count']} / {p2['pair_count']}",
        ),
        (
            "Pair Accuracy",
            f"{p1['pairwise_accuracy'] * 100:.2f}%",
            f"{p2['pairwise_accuracy'] * 100:.2f}%",
        ),
        (
            "Pairwise NLL",
            format_metric(
                p1["pairwise_nll"],
                4,
            ),
            format_metric(
                p2["pairwise_nll"],
                4,
            ),
        ),
        (
            "平均 Margin",
            format_metric(
                p1["mean_margin"],
                4,
            ),
            format_metric(
                p2["mean_margin"],
                4,
            ),
        ),
        (
            "中位 Margin",
            format_metric(
                p1["median_margin"],
                4,
            ),
            format_metric(
                p2["median_margin"],
                4,
            ),
        ),
        (
            "最小 Margin",
            format_metric(
                p1["minimum_margin"],
                4,
            ),
            format_metric(
                p2["minimum_margin"],
                4,
            ),
        ),
        (
            "当前定位",
            "主 Reward Model",
            "实验备选模型",
        ),
    ]

    start_y = 325
    row_height = 78

    for index, values in enumerate(rows):
        y1 = start_y + index * row_height
        y2 = y1 + row_height - 8

        for column_index, (
            value,
            (_, x1, x2),
        ) in enumerate(
            zip(values, columns)
        ):
            if index == len(rows) - 1:
                fill = (
                    "#F0F2F5"
                    if column_index == 0
                    else (
                        LIGHT_GREEN
                        if column_index == 1
                        else LIGHT_YELLOW
                    )
                )
            else:
                fill = (
                    "#F0F2F5"
                    if column_index == 0
                    else CARD
                )

            rounded_card(
                draw,
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
                fill=fill,
                outline=BG,
                radius=10,
                width=1,
            )

            text_color = TEXT

            if (
                values[0] == "最小 Margin"
                and column_index == 2
            ):
                text_color = BAD

            if (
                values[0] == "当前定位"
                and column_index == 1
            ):
                text_color = GOOD

            draw_text(
                draw,
                (
                    (x1 + x2) // 2,
                    (y1 + y2) // 2,
                ),
                value,
                selected_font=(
                    BODY_FONT
                    if column_index
                    else SMALL_FONT
                ),
                fill=text_color,
                anchor="mm",
            )

    rounded_card(
        draw,
        (
            70,
            895,
            1730,
            955,
        ),
        fill=LIGHT_GREEN,
        outline=LIGHT_GREEN,
        radius=16,
    )

    draw_text(
        draw,
        (95, 925),
        (
            "结论：当前小数据条件下，P1 更稳定；"
            "P2 Margin 更大，但出现一次高置信反向排序。"
        ),
        selected_font=SMALL_FONT,
        fill=TEXT,
        anchor="lm",
    )

    canvas.save(
        OUTPUT_DIR
        / "04_P1与P2正式结果对比.png",
        quality=96,
    )


def create_case_row(
    *,
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    row: dict[str, Any],
    p1_row: dict[str, Any],
    y: int,
    title: str,
):
    draw_text(
        draw,
        (55, y + 20),
        title,
        selected_font=SUBTITLE_FONT,
        fill=ACCENT,
    )

    draw_text(
        draw,
        (1680, y + 25),
        (
            f"Pair={row['pair_id']}    "
            f"P1 Margin="
            f"{float(p1_row['pairwise_margin']):+.4f}"
        ),
        selected_font=SMALL_FONT,
        fill=MUTED,
        anchor="ra",
    )

    paths = [
        (
            "R",
            resolve_image_path(
                row["reference_image_path"]
            ),
            ACCENT,
        ),
        (
            "GOOD",
            resolve_image_path(
                row["positive_image_path"]
            ),
            GOOD,
        ),
        (
            "BAD",
            resolve_image_path(
                row["negative_image_path"]
            ),
            BAD,
        ),
    ]

    panel_width = 500
    panel_height = 285
    start_x = 55
    gap = 40

    for index, (
        label,
        path,
        color,
    ) in enumerate(paths):
        x = start_x + index * (
            panel_width + gap
        )

        rounded_card(
            draw,
            (
                x,
                y + 65,
                x + panel_width,
                y + 65 + panel_height,
            ),
        )

        image = load_fitted_image(
            path,
            (
                panel_width - 18,
                panel_height - 55,
            ),
        )

        canvas.paste(
            image,
            (
                x + 9,
                y + 111,
            ),
        )

        draw_text(
            draw,
            (
                x + panel_width // 2,
                y + 89,
            ),
            label,
            selected_font=SMALL_FONT,
            fill=color,
            anchor="mm",
        )


def generate_05_margin_cases(
    manifest_index: dict[str, dict[str, Any]],
    p1_index: dict[str, dict[str, Any]],
):
    rows = sorted(
        p1_index.values(),
        key=lambda row: float(
            row["pairwise_margin"]
        ),
    )

    hard = rows[0]
    medium = rows[len(rows) // 2]
    easy = rows[-1]

    selections = [
        (
            hard,
            "困难案例：P1 Margin 最小",
        ),
        (
            medium,
            "中等案例：P1 Margin 居中",
        ),
        (
            easy,
            "简单案例：P1 Margin 最大",
        ),
    ]

    width = 1700
    height = 1320

    canvas = Image.new(
        "RGB",
        (width, height),
        BG,
    )
    draw = ImageDraw.Draw(canvas)

    draw_text(
        draw,
        (55, 35),
        "P1 在有效测试集中的简单、中等和困难案例",
        selected_font=TITLE_FONT,
    )

    draw_wrapped(
        draw,
        (57, 100),
        (
            "这里的难度按照 P1 的 GOOD−BAD Margin "
            "在 18 个样本中的相对大小自动划分，"
            "不等同于人工难度标签。"
        ),
        max_width=1580,
        selected_font=SMALL_FONT,
        fill=MUTED,
    )

    for index, (
        prediction,
        title,
    ) in enumerate(selections):
        pair_id = str(
            prediction["pair_id"]
        )

        create_case_row(
            canvas=canvas,
            draw=draw,
            row=manifest_index[pair_id],
            p1_row=prediction,
            y=155 + index * 380,
            title=title,
        )

    canvas.save(
        OUTPUT_DIR
        / "05_P1简单中等困难案例.png",
        quality=95,
    )


def generate_06_p2_error(
    manifest_index: dict[str, dict[str, Any]],
    p1_index: dict[str, dict[str, Any]],
    p2_index: dict[str, dict[str, Any]],
):
    errors = [
        row
        for row in p2_index.values()
        if not bool(row["is_correct"])
    ]

    if not errors:
        return

    selected = min(
        errors,
        key=lambda row: float(
            row["pairwise_margin"]
        ),
    )

    pair_id = str(selected["pair_id"])
    p1_margin = float(
        p1_index[pair_id][
            "pairwise_margin"
        ]
    )
    p2_margin = float(
        selected["pairwise_margin"]
    )

    create_triptych(
        row=manifest_index[pair_id],
        output_path=(
            OUTPUT_DIR
            / "06_P2高置信错误案例.png"
        ),
        title="P2 的高置信错误案例",
        subtitle=(
            "P2 在多数样本上产生较大正 Margin，"
            "但该样本被非常自信地反向排序。"
        ),
        footer=(
            f"P1 Margin={p1_margin:+.4f}；"
            f"P2 Margin={p2_margin:+.4f}。"
            "这说明平均 Margin 大并不等同于排序更可靠。"
        ),
        extra_header=f"Pair：{pair_id}",
    )


def natural_key(value: str):
    return [
        int(part)
        if part.isdigit()
        else part.lower()
        for part in re.split(
            r"(\d+)",
            value,
        )
    ]


def generate_07_excluded_overview(
    original_index: dict[str, dict[str, Any]],
    filtered_ids: set[str],
):
    excluded_ids = sorted(
        (
            pair_id
            for pair_id in original_index
            if pair_id not in filtered_ids
        ),
        key=natural_key,
    )

    cols = 5
    cell_width = 315
    cell_height = 245
    gap = 20
    left = 50
    top = 180

    rows = math.ceil(
        len(excluded_ids) / cols
    )

    width = (
        left * 2
        + cols * cell_width
        + (cols - 1) * gap
    )
    height = (
        top
        + rows * cell_height
        + (rows - 1) * gap
        + 100
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        BG,
    )
    draw = ImageDraw.Draw(canvas)

    draw_text(
        draw,
        (50, 35),
        "被排除的测试数据总览",
        selected_font=TITLE_FONT,
    )

    draw_wrapped(
        draw,
        (52, 100),
        (
            "这些样本因 Reference 非空、任务定义不一致、"
            "UID 数据不合规或视角变化等原因被排除。"
            "本图只用于数据审计，不用于模型性能展示。"
        ),
        max_width=width - 100,
        selected_font=SMALL_FONT,
        fill=MUTED,
    )

    for index, pair_id in enumerate(
        excluded_ids
    ):
        row_index = index // cols
        col_index = index % cols

        x = left + col_index * (
            cell_width + gap
        )
        y = top + row_index * (
            cell_height + gap
        )

        rounded_card(
            draw,
            (
                x,
                y,
                x + cell_width,
                y + cell_height,
            ),
        )

        image_path = resolve_image_path(
            original_index[pair_id][
                "reference_image_path"
            ]
        )

        image = load_fitted_image(
            image_path,
            (
                cell_width - 16,
                cell_height - 55,
            ),
        )

        canvas.paste(
            image,
            (
                x + 8,
                y + 43,
            ),
        )

        label_color = (
            BAD
            if pair_id == "B_05"
            else WARNING
        )

        draw_text(
            draw,
            (
                x + cell_width // 2,
                y + 23,
            ),
            pair_id,
            selected_font=TINY_FONT,
            fill=label_color,
            anchor="mm",
        )

    canvas.save(
        OUTPUT_DIR
        / "07_被排除数据总览.png",
        quality=93,
    )


def generate_08_b05(
    original_index: dict[str, dict[str, Any]],
):
    pair_id = "B_05"

    if pair_id not in original_index:
        return

    create_triptych(
        row=original_index[pair_id],
        output_path=(
            OUTPUT_DIR
            / "08_B05视角变化案例.png"
        ),
        title="无效数据示例：房型视角发生轻微变化",
        subtitle=(
            "虽然三张图片都有完整文件，但 R、GOOD、BAD "
            "之间的相机视角或房型配准不完全一致。"
        ),
        footer=(
            "这类 Pair 应排除。否则模型可能根据相机、墙面或"
            "房间结构变化判断，而不是根据家具布局质量判断。"
        ),
        extra_header="Pair：B_05",
    )


def generate_09_collection_requirements():
    width = 1800
    height = 1250

    canvas = Image.new(
        "RGB",
        (width, height),
        BG,
    )
    draw = ImageDraw.Draw(canvas)

    draw_text(
        draw,
        (65, 40),
        "下一阶段需要收集什么样的数据",
        selected_font=TITLE_FONT,
    )

    draw_wrapped(
        draw,
        (67, 110),
        (
            "目标不是简单增加图片数量，而是增加同房间、"
            "同视角、好坏明确，并覆盖真实困难情况的布局 Pair。"
        ),
        max_width=1660,
        selected_font=BODY_FONT,
        fill=MUTED,
    )

    rounded_card(
        draw,
        (
            65,
            200,
            870,
            590,
        ),
        fill=LIGHT_GREEN,
        outline=LIGHT_GREEN,
    )

    draw_text(
        draw,
        (105, 235),
        "每组数据必须满足",
        selected_font=SUBTITLE_FONT,
        fill=GOOD,
    )

    valid_items = [
        "R 是真正的空房间",
        "GOOD 与 BAD 对应同一个 R",
        "房型、门窗和墙面结构一致",
        "相机位置、角度和裁剪一致",
        "GOOD 确实比 BAD 更合理",
        "差异主要来自家具布局",
    ]

    y = 305

    for item in valid_items:
        draw.ellipse(
            (
                108,
                y + 7,
                130,
                y + 29,
            ),
            fill=GOOD,
        )
        draw_text(
            draw,
            (145, y),
            item,
            selected_font=BODY_FONT,
        )
        y += 48

    rounded_card(
        draw,
        (
            930,
            200,
            1735,
            590,
        ),
        fill=LIGHT_RED,
        outline=LIGHT_RED,
    )

    draw_text(
        draw,
        (970, 235),
        "不能进入数据集",
        selected_font=SUBTITLE_FONT,
        fill=BAD,
    )

    invalid_items = [
        "R 中已经存在待布局家具",
        "GOOD 与 BAD 来自不同房间",
        "相机视角或画面裁剪变化",
        "墙体、门窗或房间比例变化",
        "只存在审美差异，无法明确好坏",
        "GOOD/BAD 带有不同水印或渲染风格",
    ]

    y = 305

    for item in invalid_items:
        draw.line(
            (
                1080 - 100,
                y + 8,
                1080 - 78,
                y + 30,
            ),
            fill=BAD,
            width=5,
        )
        draw.line(
            (
                1080 - 78,
                y + 8,
                1080 - 100,
                y + 30,
            ),
            fill=BAD,
            width=5,
        )
        draw_text(
            draw,
            (1015, y),
            item,
            selected_font=BODY_FONT,
        )
        y += 48

    draw_text(
        draw,
        (65, 650),
        "重点补充的 BAD 类型",
        selected_font=SUBTITLE_FONT,
        fill=ACCENT,
    )

    categories = [
        (
            "碰撞与边界",
            "家具穿墙、互相穿插、悬空、尺寸不匹配",
        ),
        (
            "通行与功能",
            "挡门、挡窗、通道过窄、椅子或柜门无法使用",
        ),
        (
            "朝向关系",
            "床、沙发、电视、书桌等朝向或组合不合理",
        ),
        (
            "空间平衡",
            "过密、过空、家具堆在一侧、功能区不协调",
        ),
        (
            "困难负样本",
            "两种方案都基本合理，只在间距、动线或功能上有细微差异",
        ),
        (
            "对抗样本",
            "视觉上整洁但实际不可用，或模型高分但人工认为较差",
        ),
    ]

    card_width = 530
    card_height = 205
    category_positions = [
        (65, 715),
        (635, 715),
        (1205, 715),
        (65, 950),
        (635, 950),
        (1205, 950),
    ]

    fills = [
        LIGHT_BLUE,
        LIGHT_GREEN,
        LIGHT_YELLOW,
        "#F1ECF7",
        "#EAF6F8",
        LIGHT_RED,
    ]

    colors = [
        ACCENT,
        GOOD,
        WARNING,
        ACCENT_2,
        "#247985",
        BAD,
    ]

    for (
        heading,
        body,
    ), (
        x,
        y,
    ), fill, color in zip(
        categories,
        category_positions,
        fills,
        colors,
    ):
        rounded_card(
            draw,
            (
                x,
                y,
                x + card_width,
                y + card_height,
            ),
            fill=fill,
            outline=fill,
        )

        draw_text(
            draw,
            (x + 30, y + 28),
            heading,
            selected_font=SUBTITLE_FONT,
            fill=color,
        )

        draw_wrapped(
            draw,
            (x + 30, y + 88),
            body,
            max_width=card_width - 60,
            selected_font=SMALL_FONT,
            fill=TEXT,
        )

    canvas.save(
        OUTPUT_DIR
        / "09_数据收集规范与目标类型.png",
        quality=96,
    )


def write_readme(
    original_count: int,
    filtered_count: int,
):
    content = f"""# 对接文档插图使用说明

生成目录：

`{OUTPUT_DIR}`

项目软链接：

`/root/qwen_pref_reward/reports/stakeholder_figures`

## 图片与建议粘贴位置

### 01_任务定义_R_GOOD_BAD.png

放在“我们要解决什么问题”之后。

用途：让对接人员第一眼理解 R、GOOD、BAD 的含义。

### 02_早期用户反馈噪声说明.png

放在“为什么不再只依赖用户点赞和点踩”之后。

用途：解释误点、主观偏好、模型错误和异常数据。

注意：这是概括示意图，不是真实样本证据图。

### 03_Pointwise与Pairwise阶段结果.png

放在“技术路线变化”或“当前阶段成果”部分。

用途：展示从 Pointwise 66/100 到专项 Pairwise 的阶段变化。

注意：两阶段测试集不同，不能宣称为严格同条件提升。

### 04_P1与P2正式结果对比.png

放在“当前模型选择”部分。

用途：解释为什么当前选择 P1，而不是 P2。

### 05_P1简单中等困难案例.png

放在“模型已经能做什么”之后。

用途：展示 P1 在有效测试集中 Margin 最大、中间和最小的三个案例。

难度是按模型 Margin 相对划分，不是人工难度标签。

### 06_P2高置信错误案例.png

放在“P1 和 P2 的区别”之后。

用途：说明 P2 虽然平均 Margin 大，但可能非常自信地判断错误。

### 07_被排除数据总览.png

放在“数据质量检查”部分。

用途：说明原始完整文件并不等于有效 Pair。

原始 Pair 数：{original_count}

筛选后 Pair 数：{filtered_count}

### 08_B05视角变化案例.png

放在“哪些数据不能使用”部分。

用途：展示轻微视角变化也会破坏任务定义。

### 09_数据收集规范与目标类型.png

放在文档最后的“需要协作方提供什么数据”部分。

用途：可以直接作为数据收集人员的检查清单。

## 推荐最小对接版本

篇幅较短时，优先使用：

1. 01_任务定义_R_GOOD_BAD.png
2. 04_P1与P2正式结果对比.png
3. 05_P1简单中等困难案例.png
4. 08_B05视角变化案例.png
5. 09_数据收集规范与目标类型.png

## 当前正式结论

- 主模型：Qwen3-VL-8B + P1 A/B Logit
- 固定验证集：18 / 20
- 筛选后的任务一致性测试集：18 / 18
- P2 测试结果：17 / 18
- 当前重点：扩充严格配准、同房间同视角的困难 Pair
- 当前不优先更换 32B
"""

    (
        OUTPUT_DIR
        / "README_图片使用说明.md"
    ).write_text(
        content,
        encoding="utf-8",
    )


def validate_inputs():
    required = [
        ORIGINAL_MANIFEST,
        FILTERED_MANIFEST,
        P1_PREDICTIONS,
        P2_PREDICTIONS,
        P1_METRICS,
        P2_METRICS,
    ]

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "缺少以下输入文件：\n"
            + "\n".join(missing)
        )


def main():
    validate_inputs()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_rows = read_jsonl(
        ORIGINAL_MANIFEST
    )
    filtered_rows = read_jsonl(
        FILTERED_MANIFEST
    )

    p1_rows = read_jsonl(
        P1_PREDICTIONS
    )
    p2_rows = read_jsonl(
        P2_PREDICTIONS
    )

    p1_metrics = read_json(
        P1_METRICS
    )
    p2_metrics = read_json(
        P2_METRICS
    )

    original_index = {
        str(row["pair_id"]): row
        for row in original_rows
    }
    filtered_index = {
        str(row["pair_id"]): row
        for row in filtered_rows
    }
    p1_index = {
        str(row["pair_id"]): row
        for row in p1_rows
    }
    p2_index = {
        str(row["pair_id"]): row
        for row in p2_rows
    }

    common_ids = (
        set(filtered_index)
        & set(p1_index)
        & set(p2_index)
    )

    if len(common_ids) != 18:
        raise RuntimeError(
            "Manifest 与预测结果交集不是 18 Pair："
            f"{len(common_ids)}"
        )

    generate_01_task_definition(
        filtered_rows
    )
    generate_02_noise_explanation()
    generate_03_stage_results()
    generate_04_model_comparison(
        p1_metrics,
        p2_metrics,
    )
    generate_05_margin_cases(
        filtered_index,
        p1_index,
    )
    generate_06_p2_error(
        filtered_index,
        p1_index,
        p2_index,
    )
    generate_07_excluded_overview(
        original_index,
        set(filtered_index),
    )
    generate_08_b05(
        original_index
    )
    generate_09_collection_requirements()

    write_readme(
        original_count=len(
            original_rows
        ),
        filtered_count=len(
            filtered_rows
        ),
    )

    summary = {
        "output_dir": str(
            OUTPUT_DIR
        ),
        "font_file": FONT_FILE,
        "original_pairs": len(
            original_rows
        ),
        "filtered_pairs": len(
            filtered_rows
        ),
        "generated_files": sorted(
            path.name
            for path in OUTPUT_DIR.iterdir()
            if path.is_file()
        ),
    }

    (
        OUTPUT_DIR
        / "generation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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
