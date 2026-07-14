#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""生成按 A/B/C/D 分类的 OOF 会议展示图集。"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps


CATEGORY_INFO: Dict[str, Dict[str, str]] = {
    "A": {
        "directory": "A_label_error",
        "title": "A：原始标签疑似错误",
        "description": "人工判断原始用户标签明显不合理，可能是误点或错标。",
    },
    "B": {
        "directory": "B_subjective_preference",
        "title": "B：合理的主观偏好",
        "description": "原始标签可以理解，但更像个体审美或少数偏好。",
    },
    "C": {
        "directory": "C_model_error",
        "title": "C：模型判断错误",
        "description": "原始用户标签更合理，OOF 模型给出了错误判断。",
    },
    "D": {
        "directory": "D_uncertain_or_abnormal",
        "title": "D：无法判断或数据异常",
        "description": "信息不足、图片异常，或无法可靠判断标签是否合理。",
    },
}


def _find_font_path() -> Path | None:
    """寻找常见中文字体。"""

    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]

    for path in candidates:
        if path.is_file():
            return path

    return None


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载展示字体。"""

    font_path = _find_font_path()

    if font_path is None:
        return ImageFont.load_default()

    return ImageFont.truetype(
        str(font_path),
        size=size,
    )


def read_audit_csv(path: Path) -> List[Dict[str, Any]]:
    """读取已经填写完成的审计 CSV。"""

    if not path.is_file():
        raise FileNotFoundError(
            f"审计 CSV 不存在：{path}"
        )

    rows: List[Dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_fields = {
            "sample_id",
            "fold",
            "label",
            "label_confidence",
            "reward_score",
            "empty_room_image",
            "generated_furniture_image",
            "review_decision",
            "reviewer_notes",
        }
        fields = set(reader.fieldnames or [])
        missing = required_fields - fields

        if missing:
            raise KeyError(
                f"审计 CSV 缺少字段：{sorted(missing)}"
            )

        for line_number, source in enumerate(
            reader,
            start=2,
        ):
            sample_id = str(
                source.get("sample_id", "")
            ).strip()
            decision = str(
                source.get("review_decision", "")
            ).strip().upper()

            if not sample_id:
                raise ValueError(
                    f"sample_id 为空：{path}:{line_number}"
                )

            if sample_id in seen_sample_ids:
                raise RuntimeError(
                    f"sample_id 重复：{sample_id}"
                )

            if decision not in CATEGORY_INFO:
                raise ValueError(
                    "review_decision 必须为 A/B/C/D："
                    f"{path}:{line_number}"
                )

            seen_sample_ids.add(sample_id)

            row = dict(source)
            row["sample_id"] = sample_id
            row["review_decision"] = decision
            row["fold"] = int(source["fold"])
            row["label"] = int(source["label"])
            row["label_confidence"] = float(
                source["label_confidence"]
            )
            row["reward_score"] = float(
                source["reward_score"]
            )
            row["p_like_prior_corrected"] = float(
                source.get(
                    "p_like_prior_corrected",
                    0.0,
                )
            )
            rows.append(row)

    if not rows:
        raise RuntimeError(
            "审计 CSV 没有数据"
        )

    return rows


def _fit_image(
    image: Image.Image,
    size: Tuple[int, int],
) -> Image.Image:
    """等比例缩放并放入固定画布。"""

    target_width, target_height = size
    converted = image.convert("RGB")

    fitted = ImageOps.contain(
        converted,
        (target_width, target_height),
        method=Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (target_width, target_height),
        "white",
    )
    x = (target_width - fitted.width) // 2
    y = (target_height - fitted.height) // 2
    canvas.paste(fitted, (x, y))

    return canvas


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: Tuple[int, int],
    font: ImageFont.ImageFont,
    max_width: int,
    line_spacing: int = 8,
    fill: str = "black",
) -> int:
    """按像素宽度换行绘制文本，返回结束 y 坐标。"""

    x, y = xy
    paragraphs = str(text).splitlines() or [""]

    for paragraph in paragraphs:
        if not paragraph:
            y += font.size + line_spacing
            continue

        current = ""

        for character in paragraph:
            candidate = current + character
            box = draw.textbbox(
                (0, 0),
                candidate,
                font=font,
            )
            width = box[2] - box[0]

            if current and width > max_width:
                draw.text(
                    (x, y),
                    current,
                    font=font,
                    fill=fill,
                )
                y += font.size + line_spacing
                current = character
            else:
                current = candidate

        if current:
            draw.text(
                (x, y),
                current,
                font=font,
                fill=fill,
            )
            y += font.size + line_spacing

    return y


def build_triptych(
    row: Mapping[str, Any],
    output_path: Path,
    width: int = 1800,
    height: int = 720,
) -> None:
    """生成单条样本三联图。"""

    empty_path = Path(
        str(row["empty_room_image"])
    )
    generated_path = Path(
        str(row["generated_furniture_image"])
    )

    if not empty_path.is_file():
        raise FileNotFoundError(
            f"空房间图片不存在：{empty_path}"
        )

    if not generated_path.is_file():
        raise FileNotFoundError(
            f"生成图片不存在：{generated_path}"
        )

    margin = 24
    gap = 20
    panel_width = (
        width - 2 * margin - 2 * gap
    ) // 3
    panel_height = height - 2 * margin - 52

    canvas = Image.new(
        "RGB",
        (width, height),
        "#f2f3f5",
    )
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(26)
    body_font = load_font(22)
    small_font = load_font(18)

    with Image.open(empty_path) as empty_image:
        empty_panel = _fit_image(
            empty_image,
            (panel_width, panel_height),
        )

    with Image.open(generated_path) as generated_image:
        generated_panel = _fit_image(
            generated_image,
            (panel_width, panel_height),
        )

    left_x = margin
    middle_x = margin + panel_width + gap
    right_x = margin + 2 * (panel_width + gap)
    image_y = margin + 38

    canvas.paste(
        empty_panel,
        (left_x, image_y),
    )
    canvas.paste(
        generated_panel,
        (middle_x, image_y),
    )

    draw.text(
        (left_x, margin),
        "原始空房间",
        font=title_font,
        fill="black",
    )
    draw.text(
        (middle_x, margin),
        "生成家具结果",
        font=title_font,
        fill="black",
    )

    draw.rounded_rectangle(
        (
            right_x,
            image_y,
            right_x + panel_width,
            image_y + panel_height,
        ),
        radius=18,
        fill="white",
        outline="#d9dce1",
        width=2,
    )

    decision = str(
        row["review_decision"]
    )
    info = CATEGORY_INFO[decision]
    label_name = (
        "点赞"
        if int(row["label"]) == 1
        else "点踩"
    )
    model_prediction = (
        "点赞"
        if float(
            row["p_like_prior_corrected"]
        ) >= 0.5
        else "点踩"
    )

    x = right_x + 28
    y = image_y + 28
    text_width = panel_width - 56

    y = _draw_wrapped_text(
        draw,
        info["title"],
        (x, y),
        title_font,
        text_width,
        line_spacing=10,
    )
    y += 12

    lines = [
        f"审计序号：{row.get('audit_rank', '')}",
        f"Fold：{row['fold']}",
        f"原始标签：{label_name}",
        f"模型预测：{model_prediction}",
        f"标签可信度：{float(row['label_confidence']):.4f}",
        f"Reward：{float(row['reward_score']):.4f}",
        f"Sample ID：{str(row['sample_id'])[:16]}…",
    ]

    for line in lines:
        draw.text(
            (x, y),
            line,
            font=body_font,
            fill="black",
        )
        y += body_font.size + 12

    y += 8
    draw.text(
        (x, y),
        "人工备注：",
        font=body_font,
        fill="black",
    )
    y += body_font.size + 10

    notes = str(
        row.get("reviewer_notes", "")
    ).strip() or "未填写备注"

    _draw_wrapped_text(
        draw,
        notes,
        (x, y),
        small_font,
        text_width,
        line_spacing=7,
        fill="#333333",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    canvas.save(
        output_path,
        format="JPEG",
        quality=92,
        optimize=True,
    )


def build_overview(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """生成会议总览页。"""

    counts = Counter(
        str(row["review_decision"])
        for row in rows
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "#f4f5f7",
    )
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(52)
    subtitle_font = load_font(30)
    category_font = load_font(32)
    body_font = load_font(24)

    draw.text(
        (90, 70),
        "OOF 强冲突样本人工审计结果",
        font=title_font,
        fill="black",
    )
    draw.text(
        (90, 145),
        f"共 {len(rows)} 组，按人工结论分为 A/B/C/D 四类",
        font=subtitle_font,
        fill="#333333",
    )

    card_width = 820
    card_height = 170
    positions = [
        (90, 250),
        (1010, 250),
        (90, 470),
        (1010, 470),
    ]

    for decision, position in zip(
        ("A", "B", "C", "D"),
        positions,
    ):
        x, y = position
        info = CATEGORY_INFO[decision]
        count = counts[decision]
        percentage = (
            count / len(rows) * 100.0
        )

        draw.rounded_rectangle(
            (
                x,
                y,
                x + card_width,
                y + card_height,
            ),
            radius=22,
            fill="white",
            outline="#d9dce1",
            width=2,
        )

        draw.text(
            (x + 28, y + 22),
            f"{decision}  {count} 组（{percentage:.1f}%）",
            font=category_font,
            fill="black",
        )

        _draw_wrapped_text(
            draw,
            info["description"],
            (x + 28, y + 78),
            body_font,
            card_width - 56,
            line_spacing=8,
            fill="#333333",
        )

    category_conclusions = {
        "A": (
            "原始标签疑似错误占比最高，说明这批强冲突样本中，"
            "用户误点或标签噪声是主要问题。"
        ),
        "B": (
            "合理的主观偏好占比最高，说明用户审美差异和少数偏好"
            "是这批强冲突样本的主要来源。"
        ),
        "C": (
            "模型判断错误占比最高，说明当前 OOF 模型本身仍有较多误判，"
            "不适合直接用于自动清洗标签。"
        ),
        "D": (
            "无法判断或数据异常占比最高，说明数据质量、任务边界"
            "或信息完整性是主要问题。"
        ),
    }

    maximum_count = max(
        counts[decision]
        for decision in ("A", "B", "C", "D")
    )
    leading_categories = [
        decision
        for decision in ("A", "B", "C", "D")
        if counts[decision] == maximum_count
    ]

    if len(leading_categories) == 1:
        leading = leading_categories[0]
        conclusion = (
            f"主要结论：{leading} 类最多，共 {maximum_count} 组。"
            + category_conclusions[leading]
        )
    else:
        joined = "、".join(leading_categories)
        conclusion = (
            f"主要结论：{joined} 类并列最多，各 {maximum_count} 组。"
            "这说明强冲突样本存在多种并行原因，不能只归因于单一问题。"
        )

    draw.rounded_rectangle(
        (90, 740, 1830, 970),
        radius=22,
        fill="white",
        outline="#d9dce1",
        width=2,
    )
    _draw_wrapped_text(
        draw,
        conclusion,
        (130, 790),
        category_font,
        1660,
        line_spacing=14,
        fill="black",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    canvas.save(
        output_path,
        format="JPEG",
        quality=94,
        optimize=True,
    )


def build_contact_sheets(
    category_rows: Sequence[Mapping[str, Any]],
    triptych_paths: Sequence[Path],
    output_dir: Path,
    decision: str,
    items_per_sheet: int = 2,
    width: int = 1920,
    height: int = 1080,
) -> List[Path]:
    """把三联图拼成 16:9 会议展示页。"""

    if items_per_sheet <= 0:
        raise ValueError(
            "items_per_sheet 必须大于 0"
        )

    if len(category_rows) != len(triptych_paths):
        raise ValueError(
            "category_rows 与 triptych_paths 数量不一致"
        )

    output_paths: List[Path] = []
    page_count = math.ceil(
        len(triptych_paths)
        / items_per_sheet
    )

    title_font = load_font(34)
    small_font = load_font(22)

    for page_index in range(page_count):
        start = page_index * items_per_sheet
        end = min(
            start + items_per_sheet,
            len(triptych_paths),
        )

        page = Image.new(
            "RGB",
            (width, height),
            "#f4f5f7",
        )
        draw = ImageDraw.Draw(page)

        info = CATEGORY_INFO[decision]
        draw.text(
            (50, 28),
            (
                f"{info['title']} "
                f"（{page_index + 1}/{page_count}）"
            ),
            font=title_font,
            fill="black",
        )

        content_top = 88
        content_bottom = height - 28
        available_height = (
            content_bottom - content_top
        )
        slot_height = (
            available_height // items_per_sheet
        )

        for local_index, source_path in enumerate(
            triptych_paths[start:end]
        ):
            with Image.open(source_path) as image:
                fitted = ImageOps.contain(
                    image.convert("RGB"),
                    (
                        width - 100,
                        slot_height - 22,
                    ),
                    method=Image.Resampling.LANCZOS,
                )

            x = (width - fitted.width) // 2
            y = (
                content_top
                + local_index * slot_height
                + 6
            )
            page.paste(fitted, (x, y))

        output_path = (
            output_dir
            / f"{decision}_{page_index + 1:02d}.jpg"
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        page.save(
            output_path,
            format="JPEG",
            quality=92,
            optimize=True,
        )
        output_paths.append(output_path)

    return output_paths


def write_manifest_csv(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """写出会议展示清单。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "review_decision",
        "category_title",
        "category_rank",
        "audit_rank",
        "fold",
        "label",
        "sample_id",
        "label_confidence",
        "reward_score",
        "reviewer_notes",
        "triptych_path",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fields
                }
            )


def build_html_gallery(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """生成分类 HTML 图集。"""

    sections: List[str] = []

    for decision in ("A", "B", "C", "D"):
        info = CATEGORY_INFO[decision]
        category_rows = [
            row
            for row in rows
            if row["review_decision"]
            == decision
        ]

        cards = []

        for row in category_rows:
            cards.append(
                f"""
<article class="card">
  <img src="{html.escape(str(row['triptych_path']))}" loading="lazy">
  <div class="caption">
    #{int(row['category_rank'])} · Fold {int(row['fold'])} ·
    confidence={float(row['label_confidence']):.4f}
  </div>
</article>
"""
            )

        sections.append(
            f"""
<section>
  <h2>{html.escape(info['title'])}（{len(category_rows)}）</h2>
  <p>{html.escape(info['description'])}</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</section>
"""
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OOF 审计会议图集</title>
<style>
body {{
  margin: 0;
  padding: 28px;
  font-family: sans-serif;
  background: #f4f5f7;
}}
h1, h2 {{
  margin-top: 0;
}}
section {{
  margin: 28px 0 44px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}}
.card {{
  background: white;
  border-radius: 10px;
  overflow: hidden;
}}
.card img {{
  width: 100%;
  display: block;
}}
.caption {{
  padding: 10px 14px;
}}
@media (max-width: 1000px) {{
  .grid {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<h1>OOF 强冲突样本人工审计会议图集</h1>
{''.join(sections)}
</body>
</html>
"""

    output_path.write_text(
        document,
        encoding="utf-8",
    )


def build_meeting_gallery(
    audit_csv: Path,
    output_dir: Path,
    items_per_sheet: int = 2,
) -> Dict[str, Any]:
    """生成完整会议展示目录。"""

    rows = read_audit_csv(
        audit_csv
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overview_path = (
        output_dir
        / "contact_sheets"
        / "00_overview.jpg"
    )
    build_overview(
        rows,
        overview_path,
    )

    enriched_rows: List[
        Dict[str, Any]
    ] = []
    contact_sheet_paths = [
        overview_path
    ]

    for decision in ("A", "B", "C", "D"):
        info = CATEGORY_INFO[decision]
        category_rows = sorted(
            [
                dict(row)
                for row in rows
                if row["review_decision"]
                == decision
            ],
            key=lambda row: (
                float(row["label_confidence"]),
                int(row.get("audit_rank", 0) or 0),
                str(row["sample_id"]),
            ),
        )

        triptych_paths: List[Path] = []

        for category_rank, row in enumerate(
            category_rows,
            start=1,
        ):
            short_id = str(
                row["sample_id"]
            )[:12]
            filename = (
                f"{category_rank:02d}_"
                f"fold{int(row['fold'])}_"
                f"{short_id}.jpg"
            )
            relative_path = (
                Path(info["directory"])
                / filename
            )
            output_path = (
                output_dir
                / relative_path
            )

            build_triptych(
                row,
                output_path,
            )

            row["category_rank"] = (
                category_rank
            )
            row["category_title"] = (
                info["title"]
            )
            row["triptych_path"] = (
                relative_path.as_posix()
            )

            enriched_rows.append(row)
            triptych_paths.append(
                output_path
            )

        contact_sheet_paths.extend(
            build_contact_sheets(
                category_rows=category_rows,
                triptych_paths=triptych_paths,
                output_dir=(
                    output_dir
                    / "contact_sheets"
                ),
                decision=decision,
                items_per_sheet=(
                    items_per_sheet
                ),
            )
        )

    write_manifest_csv(
        enriched_rows,
        output_dir
        / "meeting_manifest.csv",
    )
    build_html_gallery(
        enriched_rows,
        output_dir / "index.html",
    )

    counts = Counter(
        row["review_decision"]
        for row in rows
    )

    summary = {
        "audit_csv": str(
            audit_csv.resolve()
        ),
        "output_dir": str(
            output_dir.resolve()
        ),
        "total_samples": len(rows),
        "category_counts": {
            decision: int(
                counts[decision]
            )
            for decision in (
                "A",
                "B",
                "C",
                "D",
            )
        },
        "outputs": {
            "overview": str(
                overview_path.resolve()
            ),
            "html_gallery": str(
                (
                    output_dir
                    / "index.html"
                ).resolve()
            ),
            "manifest": str(
                (
                    output_dir
                    / "meeting_manifest.csv"
                ).resolve()
            ),
            "contact_sheets_dir": str(
                (
                    output_dir
                    / "contact_sheets"
                ).resolve()
            ),
        },
        "contact_sheet_count": len(
            contact_sheet_paths
        ),
    }

    (
        output_dir
        / "meeting_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return summary
