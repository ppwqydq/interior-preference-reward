#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""构建分 Fold、分类别的 OOF 人工审计集。"""

from __future__ import annotations

import csv
import html
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)


_REQUIRED_FIELDS = {
    "sample_id",
    "fold",
    "label",
    "label_confidence",
    "conflict_score",
    "reward_score",
    "p_like_raw",
    "p_like_prior_corrected",
    "empty_room_image",
    "generated_furniture_image",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 对象列表。"""

    if not path.is_file():
        raise FileNotFoundError(
            f"JSONL 文件不存在：{path}"
        )

    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
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
                    f"JSONL 解析失败："
                    f"{path}:{line_number}"
                ) from exc

            if not isinstance(row, dict):
                raise TypeError(
                    f"JSONL 行必须是对象："
                    f"{path}:{line_number}"
                )

            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"JSONL 文件为空：{path}"
        )

    return rows


def normalize_prediction(
    source: Mapping[str, Any],
) -> Dict[str, Any]:
    """校验并标准化一条 OOF 预测。"""

    row = dict(source)
    missing_fields = (
        _REQUIRED_FIELDS - set(row)
    )

    if missing_fields:
        raise KeyError(
            "OOF 预测缺少字段："
            f"{sorted(missing_fields)}"
        )

    sample_id = str(
        row["sample_id"]
    ).strip()

    if not sample_id:
        raise ValueError(
            "sample_id 不能为空"
        )

    fold = int(row["fold"])

    if fold <= 0:
        raise ValueError(
            f"fold 必须大于 0：{sample_id}"
        )

    label = int(row["label"])

    if label not in (0, 1):
        raise ValueError(
            f"label 必须为 0 或 1：{sample_id}"
        )

    label_confidence = float(
        row["label_confidence"]
    )
    conflict_score = float(
        row["conflict_score"]
    )
    p_like_raw = float(
        row["p_like_raw"]
    )
    p_like_corrected = float(
        row["p_like_prior_corrected"]
    )

    for field_name, value in {
        "label_confidence": label_confidence,
        "conflict_score": conflict_score,
        "p_like_raw": p_like_raw,
        "p_like_prior_corrected": (
            p_like_corrected
        ),
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{field_name} 不在 [0, 1]："
                f"sample_id={sample_id}，"
                f"value={value}"
            )

    return {
        **row,
        "sample_id": sample_id,
        "fold": fold,
        "label": label,
        "label_confidence": (
            label_confidence
        ),
        "conflict_score": (
            conflict_score
        ),
        "reward_score": float(
            row["reward_score"]
        ),
        "p_like_raw": p_like_raw,
        "p_like_prior_corrected": (
            p_like_corrected
        ),
        "empty_room_image": str(
            row["empty_room_image"]
        ),
        "generated_furniture_image": str(
            row["generated_furniture_image"]
        ),
    }


def select_balanced_audit_samples(
    predictions: Iterable[Mapping[str, Any]],
    per_fold_per_label: int,
) -> List[Dict[str, Any]]:
    """按 Fold 和原标签等额选择最低可信度样本。"""

    if per_fold_per_label <= 0:
        raise ValueError(
            "per_fold_per_label 必须大于 0"
        )

    groups: Dict[
        tuple[int, int],
        List[Dict[str, Any]],
    ] = defaultdict(list)

    seen_sample_ids: set[str] = set()

    for source in predictions:
        row = normalize_prediction(source)
        sample_id = str(row["sample_id"])

        if sample_id in seen_sample_ids:
            raise RuntimeError(
                f"sample_id 重复：{sample_id}"
            )

        seen_sample_ids.add(sample_id)

        groups[
            (
                int(row["fold"]),
                int(row["label"]),
            )
        ].append(row)

    if not groups:
        raise RuntimeError(
            "没有可供审计的 OOF 预测"
        )

    folds = sorted(
        {
            fold
            for fold, _ in groups
        }
    )

    selected: List[Dict[str, Any]] = []

    for fold in folds:
        for label in (0, 1):
            key = (fold, label)
            candidates = sorted(
                groups.get(key, []),
                key=lambda row: (
                    float(
                        row[
                            "label_confidence"
                        ]
                    ),
                    str(row["sample_id"]),
                ),
            )

            if len(candidates) < (
                per_fold_per_label
            ):
                raise RuntimeError(
                    "候选样本不足："
                    f"fold={fold}，label={label}，"
                    f"required={per_fold_per_label}，"
                    f"available={len(candidates)}"
                )

            for group_rank, row in enumerate(
                candidates[
                    :per_fold_per_label
                ],
                start=1,
            ):
                selected.append(
                    {
                        **row,
                        "audit_group_rank": (
                            group_rank
                        ),
                    }
                )

    selected.sort(
        key=lambda row: (
            int(row["fold"]),
            int(row["label"]),
            int(row["audit_group_rank"]),
        )
    )

    for global_rank, row in enumerate(
        selected,
        start=1,
    ):
        row["audit_rank"] = global_rank
        row["label_name"] = (
            "like"
            if int(row["label"]) == 1
            else "dislike"
        )
        row["review_decision"] = ""
        row["reviewer_notes"] = ""

    return selected


def materialize_assets(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    asset_mode: str,
) -> List[Dict[str, Any]]:
    """把审计图片链接为相对资产路径。"""

    normalized_mode = (
        asset_mode.strip().lower()
    )

    if normalized_mode not in {
        "symlink",
        "copy",
        "none",
    }:
        raise ValueError(
            "asset_mode 必须为 "
            "symlink、copy 或 none"
        )

    if normalized_mode == "none":
        return [
            dict(row)
            for row in rows
        ]

    assets_dir = (
        output_dir / "assets"
    )
    assets_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    materialized: List[
        Dict[str, Any]
    ] = []

    for row in rows:
        copied = dict(row)
        sample_id = str(
            row["sample_id"]
        )

        source_paths = {
            "empty_room_image": Path(
                str(
                    row[
                        "empty_room_image"
                    ]
                )
            ),
            "generated_furniture_image": Path(
                str(
                    row[
                        "generated_furniture_image"
                    ]
                )
            ),
        }

        for field_name, source_path in (
            source_paths.items()
        ):
            if not source_path.is_file():
                raise FileNotFoundError(
                    "审计图片不存在："
                    f"{field_name}={source_path}"
                )

            role = (
                "empty"
                if field_name
                == "empty_room_image"
                else "generated"
            )
            suffix = (
                source_path.suffix.lower()
                or ".img"
            )
            destination = (
                assets_dir
                / f"{sample_id}_{role}{suffix}"
            )

            if destination.exists() or (
                destination.is_symlink()
            ):
                destination.unlink()

            if normalized_mode == "copy":
                shutil.copy2(
                    source_path,
                    destination,
                )
            else:
                destination.symlink_to(
                    source_path.resolve()
                )

            copied[
                f"{field_name}_audit_path"
            ] = destination.relative_to(
                output_dir
            ).as_posix()

        materialized.append(copied)

    return materialized


def write_csv_atomic(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """原子写入可填写的审计 CSV。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "audit_rank",
        "fold",
        "label",
        "label_name",
        "audit_group_rank",
        "sample_id",
        "label_confidence",
        "conflict_score",
        "reward_score",
        "p_like_raw",
        "p_like_prior_corrected",
        "empty_room_image",
        "generated_furniture_image",
        "empty_room_image_audit_path",
        "generated_furniture_image_audit_path",
        "review_decision",
        "reviewer_notes",
    ]

    file_descriptor, temp_name = (
        tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=str(output_path.parent),
        )
    )

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        field_name: row.get(
                            field_name,
                            "",
                        )
                        for field_name
                        in fieldnames
                    }
                )

            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temp_name,
            output_path,
        )

    except Exception:
        try:
            os.remove(temp_name)
        except OSError:
            pass
        raise


def build_html(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """生成双图并排的静态审计页面。"""

    cards: List[str] = []

    for row in rows:
        empty_path = str(
            row.get(
                "empty_room_image_audit_path",
                row["empty_room_image"],
            )
        )
        generated_path = str(
            row.get(
                "generated_furniture_image_audit_path",
                row[
                    "generated_furniture_image"
                ],
            )
        )

        cards.append(
            f"""
<section class="card">
  <header>
    <strong>#{int(row["audit_rank"])}</strong>
    <span>Fold {int(row["fold"])}</span>
    <span>原标签：{html.escape(str(row["label_name"]))}</span>
    <span>标签可信度：{float(row["label_confidence"]):.4f}</span>
    <span>冲突分数：{float(row["conflict_score"]):.4f}</span>
  </header>
  <div class="images">
    <figure>
      <img src="{html.escape(empty_path)}" loading="lazy">
      <figcaption>原始空房间</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(generated_path)}" loading="lazy">
      <figcaption>生成家具结果</figcaption>
    </figure>
  </div>
  <div class="meta">
    <code>{html.escape(str(row["sample_id"]))}</code>
    <div>
      reward={float(row["reward_score"]):.4f}，
      p_like_corrected={float(row["p_like_prior_corrected"]):.4f}
    </div>
    <div class="decision">
      审计结论填写到 CSV：
      A=标签明显可能错误；
      B=标签合理但偏好主观；
      C=模型判断错误；
      D=无法判断或图片异常。
    </div>
  </div>
</section>
"""
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OOF 人工审计集</title>
<style>
body {{
  margin: 0;
  padding: 24px;
  font-family: sans-serif;
  background: #f4f5f7;
}}
h1 {{
  margin-top: 0;
}}
.notice {{
  padding: 12px 16px;
  margin-bottom: 18px;
  background: white;
  border-radius: 8px;
}}
.card {{
  margin: 0 0 20px;
  padding: 16px;
  background: white;
  border-radius: 10px;
}}
header {{
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-bottom: 12px;
}}
.images {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}}
figure {{
  margin: 0;
}}
img {{
  width: 100%;
  max-height: 520px;
  object-fit: contain;
  background: #eee;
}}
figcaption {{
  margin-top: 6px;
  text-align: center;
}}
.meta {{
  margin-top: 12px;
  line-height: 1.6;
  word-break: break-all;
}}
.decision {{
  margin-top: 6px;
}}
@media (max-width: 800px) {{
  .images {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<h1>OOF 人工审计集</h1>
<div class="notice">
每个 Fold、每个原标签分别选取最低可信度样本。
请结合两张图片人工判断，并在 audit_template.csv 中填写
review_decision 和 reviewer_notes。
</div>
{''.join(cards)}
</body>
</html>
"""

    output_path.write_text(
        document,
        encoding="utf-8",
    )


def build_audit_set(
    predictions_path: Path,
    output_dir: Path,
    per_fold_per_label: int,
    asset_mode: str,
) -> Dict[str, Any]:
    """构建完整 OOF 人工审计集。"""

    predictions = read_jsonl(
        predictions_path
    )
    selected = (
        select_balanced_audit_samples(
            predictions=predictions,
            per_fold_per_label=(
                per_fold_per_label
            ),
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    materialized = materialize_assets(
        rows=selected,
        output_dir=output_dir,
        asset_mode=asset_mode,
    )

    jsonl_path = (
        output_dir
        / "audit_manifest.jsonl"
    )
    csv_path = (
        output_dir
        / "audit_template.csv"
    )
    html_path = (
        output_dir
        / "index.html"
    )
    summary_path = (
        output_dir
        / "audit_summary.json"
    )

    write_jsonl_atomic(
        materialized,
        jsonl_path,
    )
    write_csv_atomic(
        materialized,
        csv_path,
    )
    build_html(
        materialized,
        html_path,
    )

    group_counts = Counter(
        (
            int(row["fold"]),
            int(row["label"]),
        )
        for row in materialized
    )

    summary = {
        "predictions_path": str(
            predictions_path.resolve()
        ),
        "output_dir": str(
            output_dir.resolve()
        ),
        "selected_samples": len(
            materialized
        ),
        "per_fold_per_label": (
            per_fold_per_label
        ),
        "asset_mode": asset_mode,
        "group_counts": {
            f"fold_{fold}_label_{label}": (
                int(count)
            )
            for (
                fold,
                label,
            ), count in sorted(
                group_counts.items()
            )
        },
        "label_counts": {
            "0": sum(
                int(row["label"]) == 0
                for row in materialized
            ),
            "1": sum(
                int(row["label"]) == 1
                for row in materialized
            ),
        },
        "outputs": {
            "manifest": str(
                jsonl_path.resolve()
            ),
            "template_csv": str(
                csv_path.resolve()
            ),
            "html": str(
                html_path.resolve()
            ),
        },
        "review_decisions": {
            "A": "原始用户标签明显可能错误",
            "B": "原始标签合理，但属于主观或少数偏好",
            "C": "模型判断错误，原始标签更合理",
            "D": "无法判断、图片异常或任务信息不足",
        },
    }

    write_json_atomic(
        summary,
        summary_path,
    )

    return summary
