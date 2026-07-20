#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path("/root/qwen_pref_reward")
OUTPUT_ROOT = Path(
    "/root/autodl-tmp/qwen_pref_reward_outputs"
)

FIGURE_DIR = (
    OUTPUT_ROOT
    / "stakeholder_figures"
    / "current_results"
)

SIX_MODEL_EVAL_ROOT = (
    OUTPUT_ROOT
    / "dataset_nft_syn_filtered_evaluation_6models_20260719_141015"
)

SEED_ROOT = (
    OUTPUT_ROOT
    / "p1_512_seed_stability_20260719_233212"
)

DATASET_ROOT = Path(
    "/root/autodl-tmp/datasets/dataset_nft_syn"
)

EXTERNAL_MANIFEST = (
    DATASET_ROOT
    / "pairwise_test_filtered.jsonl"
)

FONT_FILE = Path(
    "/usr/share/fonts/opentype/noto/"
    "NotoSansCJK-Regular.ttc"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path} 第 {line_number} 行 JSON 无效"
                ) from exc

            rows.append(row)

    return rows


def get_number(
    row: dict[str, Any],
    *keys: str,
) -> float:
    for key in keys:
        if key in row and row[key] is not None:
            return float(row[key])

    raise KeyError(
        f"找不到字段：{keys}；现有字段：{list(row)}"
    )


font_prop = FontProperties(
    fname=str(FONT_FILE)
)

plt.rcParams["axes.unicode_minus"] = False


def apply_chinese_font(ax: Any) -> None:
    ax.title.set_fontproperties(font_prop)
    ax.xaxis.label.set_fontproperties(font_prop)
    ax.yaxis.label.set_fontproperties(font_prop)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)


def annotate_bars(
    ax: Any,
    bars: Any,
    *,
    percentage: bool = False,
) -> None:
    for bar in bars:
        value = float(bar.get_height())

        if percentage:
            label = f"{value * 100:.1f}%"
        else:
            label = f"{value:.3f}"

        ax.annotate(
            label,
            xy=(
                bar.get_x() + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontproperties=font_prop,
            fontsize=9,
        )


# ============================================================
# 1. 分辨率实验
# ============================================================

model_names = {
    ("P1", 512):
        "qwen3_vl_8b_layout100_pairwise_p1_ab_512",
    ("P1", 768):
        "qwen3_vl_8b_layout100_pairwise_p1_ab_768",
    ("P1", 1024):
        "qwen3_vl_8b_layout100_pairwise_p1_ab_1024",

    ("P2", 512):
        "qwen3_vl_8b_layout100_pairwise_p2_scalar_512",
    ("P2", 768):
        "qwen3_vl_8b_layout100_pairwise_p2_scalar_768",
    ("P2", 1024):
        "qwen3_vl_8b_layout100_pairwise_p2_scalar_1024",
}

resolution_rows: list[dict[str, Any]] = []

for reward_type in ("P1", "P2"):
    for resolution in (512, 768, 1024):
        model_name = model_names[
            reward_type,
            resolution,
        ]

        best = load_json(
            OUTPUT_ROOT
            / model_name
            / "best_checkpoint.json"
        )

        val_metrics = best["metrics"]

        external_metrics = load_json(
            SIX_MODEL_EVAL_ROOT
            / model_name
            / "metrics.json"
        )

        resolution_rows.append({
            "reward_type": reward_type,
            "resolution": resolution,
            "val_accuracy": get_number(
                val_metrics,
                "pairwise_accuracy",
                "accuracy",
            ),
            "val_nll": get_number(
                val_metrics,
                "pairwise_nll",
                "nll",
            ),
            "external_accuracy": get_number(
                external_metrics,
                "pairwise_accuracy",
                "accuracy",
            ),
            "external_nll": get_number(
                external_metrics,
                "pairwise_nll",
                "nll",
            ),
        })


def select_resolution(
    reward_type: str,
    key: str,
) -> list[float]:
    return [
        float(row[key])
        for row in resolution_rows
        if row["reward_type"] == reward_type
    ]


resolutions = [512, 768, 1024]
x = np.arange(len(resolutions))
width = 0.34

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.6),
)

for ax, metric_key, title in (
    (
        axes[0],
        "val_accuracy",
        "固定验证集",
    ),
    (
        axes[1],
        "external_accuracy",
        "外部18组有效测试集",
    ),
):
    p1_values = select_resolution(
        "P1",
        metric_key,
    )
    p2_values = select_resolution(
        "P2",
        metric_key,
    )

    bars1 = ax.bar(
        x - width / 2,
        p1_values,
        width,
        label="P1",
    )
    bars2 = ax.bar(
        x + width / 2,
        p2_values,
        width,
        label="P2",
    )

    annotate_bars(
        ax,
        bars1,
        percentage=True,
    )
    annotate_bars(
        ax,
        bars2,
        percentage=True,
    )

    ax.set_title(title)
    ax.set_xlabel("输入分辨率")
    ax.set_ylabel("Pair Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(value) for value in resolutions]
    )
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    apply_chinese_font(ax)

fig.suptitle(
    "不同输入分辨率下的布局排序准确率",
    fontproperties=font_prop,
    fontsize=16,
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "10_不同分辨率准确率对比.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.6),
)

for ax, metric_key, title in (
    (
        axes[0],
        "val_nll",
        "固定验证集",
    ),
    (
        axes[1],
        "external_nll",
        "外部18组有效测试集",
    ),
):
    p1_values = select_resolution(
        "P1",
        metric_key,
    )
    p2_values = select_resolution(
        "P2",
        metric_key,
    )

    bars1 = ax.bar(
        x - width / 2,
        p1_values,
        width,
        label="P1",
    )
    bars2 = ax.bar(
        x + width / 2,
        p2_values,
        width,
        label="P2",
    )

    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)

    ax.set_title(title)
    ax.set_xlabel("输入分辨率")
    ax.set_ylabel("Pairwise NLL（越低越好）")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(value) for value in resolutions]
    )
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    apply_chinese_font(ax)

fig.suptitle(
    "不同输入分辨率下的 Pairwise NLL",
    fontproperties=font_prop,
    fontsize=16,
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "11_不同分辨率NLL对比.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 2. 随机种子实验
# ============================================================

seed_summary_path = (
    SEED_ROOT
    / "seed_stability_summary.csv"
)

with seed_summary_path.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:
    seed_rows = list(
        csv.DictReader(f)
    )

label_order = [
    "baseline_existing",
    "seed_17",
    "seed_29",
    "seed_43",
]

display_names = {
    "baseline_existing": "原模型",
    "seed_17": "Seed 17",
    "seed_29": "Seed 29",
    "seed_43": "Seed 43",
}

seed_index = {
    row["label"]: row
    for row in seed_rows
}

ordered_seed_rows = [
    seed_index[label]
    for label in label_order
]

seed_x = np.arange(
    len(ordered_seed_rows)
)

seed_labels = [
    display_names[row["label"]]
    for row in ordered_seed_rows
]

val_accuracy = [
    float(row["val_accuracy"])
    for row in ordered_seed_rows
]

external_accuracy = [
    float(row["external_accuracy"])
    for row in ordered_seed_rows
]

external_min_margin = [
    float(row["external_min_margin"])
    for row in ordered_seed_rows
]

fig, ax = plt.subplots(
    figsize=(10.5, 6),
)

bars1 = ax.bar(
    seed_x - width / 2,
    val_accuracy,
    width,
    label="验证集",
)

bars2 = ax.bar(
    seed_x + width / 2,
    external_accuracy,
    width,
    label="外部18组测试集",
)

annotate_bars(
    ax,
    bars1,
    percentage=True,
)

annotate_bars(
    ax,
    bars2,
    percentage=True,
)

ax.set_title(
    "P1-512 不同随机种子准确率对比"
)
ax.set_ylabel("Pair Accuracy")
ax.set_xticks(seed_x)
ax.set_xticklabels(seed_labels)
ax.set_ylim(0, 1.12)
ax.legend()
ax.grid(
    axis="y",
    alpha=0.25,
)

apply_chinese_font(ax)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "12_随机种子准确率对比.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


fig, ax = plt.subplots(
    figsize=(10.5, 6),
)

bars = ax.bar(
    seed_x,
    external_min_margin,
)

ax.axhline(
    0,
    linewidth=1.2,
)

for bar, value in zip(
    bars,
    external_min_margin,
):
    offset = 5 if value >= 0 else -18

    ax.annotate(
        f"{value:+.3f}",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            value,
        ),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va=(
            "bottom"
            if value >= 0
            else "top"
        ),
        fontproperties=font_prop,
        fontsize=10,
    )

ax.set_title(
    "不同随机种子在外部测试集上的最小 Margin"
)
ax.set_ylabel(
    "最小 GOOD-BAD Margin"
)
ax.set_xticks(seed_x)
ax.set_xticklabels(seed_labels)
ax.grid(
    axis="y",
    alpha=0.25,
)

apply_chinese_font(ax)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "13_随机种子最小Margin对比.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 3. P1-768 低学习率训练曲线
# ============================================================

log_roots = [
    PROJECT_ROOT / "logs",
    Path(
        "/root/autodl-tmp/"
        "qwen_pref_reward_logs"
    ),
    OUTPUT_ROOT,
]

target_text = (
    "qwen8b_layout100_pairwise_"
    "p1_ab_768_lr5e6.yaml"
)

lr_log: Path | None = None

for root in log_roots:
    if not root.exists():
        continue

    for path in sorted(
        root.rglob("*.log"),
        reverse=True,
    ):
        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:
                beginning = "".join(
                    f.readline()
                    for _ in range(25)
                )

            if target_text in beginning:
                lr_log = path
                break

        except OSError:
            continue

    if lr_log:
        break


epoch_pattern = re.compile(
    r"EPOCH\s+(\d+)"
    r".*?train_pair_acc=([0-9.]+)"
    r".*?val_pair_acc=([0-9.]+)"
    r".*?val_pair_nll=([0-9.]+)"
)

if lr_log:
    epochs: list[int] = []
    train_acc: list[float] = []
    val_acc: list[float] = []
    val_nll: list[float] = []

    for line in lr_log.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        match = epoch_pattern.search(line)

        if not match:
            continue

        epochs.append(
            int(match.group(1))
        )
        train_acc.append(
            float(match.group(2))
        )
        val_acc.append(
            float(match.group(3))
        )
        val_nll.append(
            float(match.group(4))
        )

    if epochs:
        fig, ax = plt.subplots(
            figsize=(11, 6),
        )

        ax.plot(
            epochs,
            train_acc,
            marker="o",
            label="训练集 Pair Accuracy",
        )
        ax.plot(
            epochs,
            val_acc,
            marker="o",
            label="验证集 Pair Accuracy",
        )

        ax.set_title(
            "P1-768 降低学习率后的训练过程"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pair Accuracy")
        ax.set_ylim(0.5, 1.02)
        ax.set_xticks(epochs)
        ax.legend()
        ax.grid(alpha=0.25)

        apply_chinese_font(ax)

        fig.tight_layout()

        fig.savefig(
            FIGURE_DIR
            / "14_P1-768低学习率训练曲线.png",
            dpi=220,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"[OK] 训练曲线日志：{lr_log}"
        )
    else:
        print(
            f"[WARN] 找到日志但未解析出 Epoch：{lr_log}"
        )
else:
    print(
        "[WARN] 未找到 P1-768 lr=5e-6 的原始日志，"
        "跳过训练曲线。"
    )


# ============================================================
# 4. 自动选择错误样例并拼接 R / GOOD / BAD
# ============================================================

def normalize_key(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]",
        "",
        value.lower(),
    )


def canonical_pair_id(value: Any) -> str:
    text = normalize_key(str(value))

    for prefix in (
        "pair",
        "sample",
        "case",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]

    return text


def get_pair_id(
    row: dict[str, Any],
    fallback: str,
) -> str:
    for key in (
        "pair_id",
        "id",
        "sample_id",
        "pair",
        "name",
    ):
        if key in row and row[key] not in (
            None,
            "",
        ):
            return str(row[key])

    return fallback


def get_margin(
    row: dict[str, Any],
) -> float:
    for key in (
        "pairwise_margin",
        "margin",
        "good_bad_margin",
        "reward_margin",
    ):
        if key in row and row[key] is not None:
            return float(row[key])

    raise KeyError(
        f"预测结果中没有 Margin 字段：{list(row)}"
    )


def prediction_map(
    path: Path,
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}

    for index, row in enumerate(rows):
        display_id = get_pair_id(
            row,
            f"row_{index}",
        )
        key = canonical_pair_id(
            display_id
        )

        result[key] = {
            "display_id": display_id,
            "margin": get_margin(row),
            "row": row,
        }

    return result


def recursive_path_value(
    obj: Any,
    aliases: set[str],
) -> str | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = normalize_key(
                str(key)
            )

            if (
                normalized in aliases
                and isinstance(value, str)
                and value
            ):
                return value

        for value in obj.values():
            found = recursive_path_value(
                value,
                aliases,
            )

            if found:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = recursive_path_value(
                value,
                aliases,
            )

            if found:
                return found

    return None


REFERENCE_ALIASES = {
    normalize_key(value)
    for value in (
        "reference",
        "reference_image",
        "reference_path",
        "ref",
        "ref_image",
        "ref_path",
        "r",
        "r_image",
        "r_path",
    )
}

GOOD_ALIASES = {
    normalize_key(value)
    for value in (
        "chosen",
        "chosen_image",
        "chosen_path",
        "good",
        "good_image",
        "good_path",
        "preferred",
        "preferred_image",
    )
}

BAD_ALIASES = {
    normalize_key(value)
    for value in (
        "rejected",
        "rejected_image",
        "rejected_path",
        "bad",
        "bad_image",
        "bad_path",
        "negative",
        "negative_image",
    )
}


def resolve_image_path(
    raw_path: str | None,
    pair_id: str,
    role: str,
) -> Path | None:
    candidates: list[Path] = []

    if raw_path:
        raw_path = raw_path.removeprefix(
            "file://"
        )

        value = Path(raw_path).expanduser()

        if value.is_absolute():
            candidates.append(value)
        else:
            candidates.extend([
                EXTERNAL_MANIFEST.parent / value,
                PROJECT_ROOT / value,
                DATASET_ROOT / value,
            ])

        candidates.append(
            DATASET_ROOT / value.name
        )

    pair_variants = {
        str(pair_id),
        canonical_pair_id(pair_id),
    }

    raw_text = str(pair_id)

    stripped = re.sub(
        r"^(pair|case)[_-]?",
        "",
        raw_text,
        flags=re.IGNORECASE,
    )

    pair_variants.add(stripped)

    suffixes = {
        "reference": (
            "_R.png",
            "_r.png",
            "_reference.png",
        ),
        "good": (
            "_good.png",
            "_GOOD.png",
            "_chosen.png",
        ),
        "bad": (
            "_bad.png",
            "_BAD.png",
            "_rejected.png",
        ),
    }[role]

    for variant in pair_variants:
        if not variant:
            continue

        for suffix in suffixes:
            candidates.append(
                DATASET_ROOT
                / f"{variant}{suffix}"
            )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


manifest_rows = read_jsonl(
    EXTERNAL_MANIFEST
)

manifest_index: dict[str, dict[str, Any]] = {}

for index, row in enumerate(manifest_rows):
    display_id = get_pair_id(
        row,
        f"row_{index}",
    )

    manifest_index[
        canonical_pair_id(display_id)
    ] = {
        "display_id": display_id,
        "row": row,
    }


def load_triplet(
    pair_key: str,
) -> tuple[
    Image.Image,
    Image.Image,
    Image.Image,
    str,
]:
    manifest_item = manifest_index.get(
        pair_key
    )

    if not manifest_item:
        raise KeyError(
            f"Manifest 中找不到 Pair：{pair_key}"
        )

    row = manifest_item["row"]
    display_id = manifest_item[
        "display_id"
    ]

    reference_path = resolve_image_path(
        recursive_path_value(
            row,
            REFERENCE_ALIASES,
        ),
        display_id,
        "reference",
    )

    good_path = resolve_image_path(
        recursive_path_value(
            row,
            GOOD_ALIASES,
        ),
        display_id,
        "good",
    )

    bad_path = resolve_image_path(
        recursive_path_value(
            row,
            BAD_ALIASES,
        ),
        display_id,
        "bad",
    )

    missing = [
        name
        for name, path in (
            ("R", reference_path),
            ("GOOD", good_path),
            ("BAD", bad_path),
        )
        if path is None
    ]

    if missing:
        raise FileNotFoundError(
            f"{display_id} 缺少图片：{missing}"
        )

    return (
        Image.open(reference_path).convert("RGB"),
        Image.open(good_path).convert("RGB"),
        Image.open(bad_path).convert("RGB"),
        str(display_id),
    )


pil_title_font = ImageFont.truetype(
    str(FONT_FILE),
    34,
)

pil_label_font = ImageFont.truetype(
    str(FONT_FILE),
    28,
)

pil_note_font = ImageFont.truetype(
    str(FONT_FILE),
    23,
)


def fit_image(
    image: Image.Image,
    width: int,
    height: int,
) -> Image.Image:
    contained = ImageOps.contain(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    x_offset = (
        width - contained.width
    ) // 2

    y_offset = (
        height - contained.height
    ) // 2

    canvas.paste(
        contained,
        (x_offset, y_offset),
    )

    return canvas


def make_error_figure(
    cases: list[
        tuple[str, str]
    ],
    title: str,
    output_name: str,
) -> None:
    cell_width = 440
    cell_height = 310
    side_margin = 30
    column_gap = 20
    header_height = 115
    note_height = 115

    total_width = (
        side_margin * 2
        + cell_width * 3
        + column_gap * 2
    )

    row_height = (
        cell_height
        + note_height
        + 55
    )

    total_height = (
        header_height
        + row_height * len(cases)
        + 25
    )

    canvas = Image.new(
        "RGB",
        (total_width, total_height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (side_margin, 25),
        title,
        fill="black",
        font=pil_title_font,
    )

    labels = [
        "R：空房间",
        "GOOD：较好布局",
        "BAD：较差布局",
    ]

    for column, label in enumerate(labels):
        x_position = (
            side_margin
            + column * (
                cell_width
                + column_gap
            )
        )

        draw.text(
            (
                x_position,
                header_height - 40,
            ),
            label,
            fill="black",
            font=pil_label_font,
        )

    for row_number, (
        pair_key,
        note,
    ) in enumerate(cases):
        reference, good, bad, display_id = (
            load_triplet(pair_key)
        )

        y_position = (
            header_height
            + row_number * row_height
        )

        for column, image in enumerate(
            (reference, good, bad)
        ):
            fitted = fit_image(
                image,
                cell_width,
                cell_height,
            )

            x_position = (
                side_margin
                + column * (
                    cell_width
                    + column_gap
                )
            )

            canvas.paste(
                fitted,
                (x_position, y_position),
            )

        note_text = (
            f"Pair：{display_id}｜{note}"
        )

        wrapped = textwrap.wrap(
            note_text,
            width=65,
        )

        draw.multiline_text(
            (
                side_margin,
                y_position
                + cell_height
                + 16,
            ),
            "\n".join(wrapped),
            fill="black",
            font=pil_note_font,
            spacing=8,
        )

    canvas.save(
        FIGURE_DIR / output_name,
        quality=95,
    )


p1_768_path = (
    SIX_MODEL_EVAL_ROOT
    / model_names["P1", 768]
    / "pairwise_predictions.jsonl"
)

p1_1024_path = (
    SIX_MODEL_EVAL_ROOT
    / model_names["P1", 1024]
    / "pairwise_predictions.jsonl"
)

p2_1024_path = (
    SIX_MODEL_EVAL_ROOT
    / model_names["P2", 1024]
    / "pairwise_predictions.jsonl"
)

p1_768_predictions = prediction_map(
    p1_768_path
)

p1_1024_predictions = prediction_map(
    p1_1024_path
)

p2_1024_predictions = prediction_map(
    p2_1024_path
)

p1_768_errors = sorted(
    (
        (pair_key, item)
        for pair_key, item
        in p1_768_predictions.items()
        if item["margin"] < 0
    ),
    key=lambda value: value[1]["margin"],
)

if p1_768_errors:
    selected = p1_768_errors[:2]

    make_error_figure(
        [
            (
                pair_key,
                (
                    "P1-768 判断错误，"
                    f"Margin={item['margin']:+.3f}。"
                    "负值表示模型把 BAD 排在 GOOD 前面。"
                ),
            )
            for pair_key, item in selected
        ],
        "P1-768 的边界错误样例",
        "15_P1-768边界错误案例.png",
    )


common_1024_errors = []

for pair_key in (
    set(p1_1024_predictions)
    & set(p2_1024_predictions)
):
    p1_margin = p1_1024_predictions[
        pair_key
    ]["margin"]

    p2_margin = p2_1024_predictions[
        pair_key
    ]["margin"]

    if p1_margin < 0 and p2_margin < 0:
        common_1024_errors.append(
            (
                pair_key,
                p1_margin,
                p2_margin,
            )
        )

common_1024_errors.sort(
    key=lambda value:
        value[1] + value[2]
)

if common_1024_errors:
    pair_key, p1_margin, p2_margin = (
        common_1024_errors[0]
    )

    make_error_figure(
        [(
            pair_key,
            (
                "P1-1024 与 P2-1024 均判断错误；"
                f"P1 Margin={p1_margin:+.3f}，"
                f"P2 Margin={p2_margin:+.3f}。"
                "两种评分方式在同一高分辨率输入上共同失败。"
            ),
        )],
        "1024 分辨率下 P1 与 P2 的共同错误",
        "16_1024共同错误案例.png",
    )


seed_compare_path = (
    SEED_ROOT
    / "external18_pairwise_seed_comparison.csv"
)

with seed_compare_path.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:
    seed_compare_rows = list(
        csv.DictReader(f)
    )


def optional_float(
    value: str | None,
) -> float | None:
    if value in (
        None,
        "",
        "None",
    ):
        return None

    return float(value)


disagreement_rows = []

for row in seed_compare_rows:
    margins = {
        "原模型": optional_float(
            row.get(
                "baseline_existing_margin"
            )
        ),
        "Seed 17": optional_float(
            row.get("seed_17_margin")
        ),
        "Seed 29": optional_float(
            row.get("seed_29_margin")
        ),
        "Seed 43": optional_float(
            row.get("seed_43_margin")
        ),
    }

    valid = [
        value
        for value in margins.values()
        if value is not None
    ]

    if not valid:
        continue

    signs = {
        value > 0
        for value in valid
    }

    if len(signs) > 1:
        disagreement_rows.append(
            (
                row,
                margins,
                min(valid),
            )
        )

disagreement_rows.sort(
    key=lambda value: value[2]
)

if disagreement_rows:
    row, margins, _ = disagreement_rows[0]

    pair_display = row["pair_id"]
    pair_key = canonical_pair_id(
        pair_display
    )

    margin_text = "；".join(
        (
            f"{name}={value:+.3f}"
            if value is not None
            else f"{name}=缺失"
        )
        for name, value
        in margins.items()
    )

    make_error_figure(
        [(
            pair_key,
            (
                "不同随机种子对同一样本判断不一致。"
                + margin_text
            ),
        )],
        "不同随机种子对同一困难样本的判断差异",
        "17_随机种子分歧案例.png",
    )


summary = {
    "output_dir": str(FIGURE_DIR),
    "generated_files": sorted(
        path.name
        for path in FIGURE_DIR.glob("*.png")
    ),
    "lr_log": (
        str(lr_log)
        if lr_log
        else None
    ),
}

(
    FIGURE_DIR
    / "generation_summary.json"
).write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(json.dumps(
    summary,
    ensure_ascii=False,
    indent=2,
))
