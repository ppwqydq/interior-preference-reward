#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用现有 Qwen P1 Reward，对 Klein 已生成候选做离线组内重排。"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger("rerank_klein_candidates")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def normalize(scores: list[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - values.mean()) / (std + 1e-6)


def call_scoring_function(
    score_reward_samples,
    *,
    backend,
    samples,
    batch_size: int,
    negative_weight: float,
):
    """按仓库当前函数签名动态传参，避免复制评分实现。"""
    signature = inspect.signature(score_reward_samples)
    available = {
        "backend": backend,
        "samples": samples,
        "batch_size": batch_size,
        "negative_weight": negative_weight,
        "logger": LOGGER,
        "validate_image_paths": True,
    }
    kwargs = {
        name: value
        for name, value in available.items()
        if name in signature.parameters
    }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and name not in kwargs
    ]
    if missing:
        raise TypeError(
            "score_reward_samples 出现未适配的必填参数："
            + ", ".join(missing)
            + f"；当前签名：{signature}"
        )
    LOGGER.info("score_reward_samples signature: %s", signature)
    return score_reward_samples(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qwen_project",
        default="/root/qwen_pref_reward",
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument(
        "--source_dir",
        default=(
            "/root/autodl-tmp/klein_generation_ranker_outputs/"
            "dev_feedback_20x6"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn_implementation", default="")
    parser.add_argument("--limit_rooms", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    qwen_project = Path(args.qwen_project).expanduser().resolve()
    ranker_project = Path(
        "/root/autodl-tmp/klein_generation_ranker"
    ).resolve()
    sys.path.insert(0, str(qwen_project / "src"))
    sys.path.insert(0, str(ranker_project))

    from contact_sheet import save_contact_sheet
    from preference_reward.inference.checkpoint import (
        load_reward_checkpoint_config,
    )
    from preference_reward.inference.qwen_ab_loader import (
        load_qwen_ab_backend,
    )
    from preference_reward.inference.scoring import (
        RewardScoringSample,
        score_reward_samples,
    )

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint = load_reward_checkpoint_config(checkpoint_dir)
    device = torch.device(args.device)
    backend = load_qwen_ab_backend(
        checkpoint=checkpoint,
        device=device,
        logger=LOGGER,
        attn_implementation=args.attn_implementation,
    )

    source_dir = Path(args.source_dir).expanduser().resolve()
    rooms_root = source_dir / "rooms"
    output_dir = source_dir / "p1_reranked"
    sheets_dir = output_dir / "contact_sheets"
    top1_dir = output_dir / "top1"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    top1_dir.mkdir(parents=True, exist_ok=True)

    room_dirs = sorted(path for path in rooms_root.iterdir() if path.is_dir())
    if args.limit_rooms > 0:
        room_dirs = room_dirs[: args.limit_rooms]
    if not room_dirs:
        raise FileNotFoundError(f"没有找到房间目录：{rooms_root}")

    scoring_samples = []
    sample_metadata: dict[str, dict[str, Any]] = {}
    room_assets: dict[str, dict[str, Any]] = {}

    for room_dir in room_dirs:
        room_id = room_dir.name
        reference_path = room_dir / "reference.png"
        candidate_paths = sorted(room_dir.glob("candidate_*.png"))
        if not reference_path.is_file() or not candidate_paths:
            LOGGER.warning("[%s] 缺少 reference 或 candidate，跳过", room_id)
            continue

        room_assets[room_id] = {
            "reference_path": reference_path,
            "candidate_paths": candidate_paths,
        }
        for candidate_index, candidate_path in enumerate(candidate_paths):
            sample_id = f"{room_id}::{candidate_index:02d}"
            scoring_samples.append(
                RewardScoringSample(
                    sample_id=sample_id,
                    empty_room_image=reference_path,
                    generated_furniture_image=candidate_path,
                    metadata={
                        "room_id": room_id,
                        "candidate_index": candidate_index,
                    },
                )
            )
            sample_metadata[sample_id] = {
                "room_id": room_id,
                "candidate_index": candidate_index,
                "candidate_path": candidate_path,
            }

    LOGGER.info(
        "准备评分：rooms=%d candidates=%d checkpoint=%s epoch=%s",
        len(room_assets),
        len(scoring_samples),
        checkpoint_dir,
        checkpoint.epoch,
    )
    rows = call_scoring_function(
        score_reward_samples,
        backend=backend,
        samples=scoring_samples,
        batch_size=args.batch_size,
        negative_weight=checkpoint.negative_weight,
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row["sample_id"])
        meta = sample_metadata[sample_id]
        grouped[meta["room_id"]].append(
            {
                "sample_id": sample_id,
                "candidate_index": meta["candidate_index"],
                "candidate_path": meta["candidate_path"],
                "reward_score": float(row["reward_score"]),
                "logit_A_like": (
                    None
                    if "logit_A_like" not in row
                    else float(row["logit_A_like"])
                ),
                "logit_B_dislike": (
                    None
                    if "logit_B_dislike" not in row
                    else float(row["logit_B_dislike"])
                ),
                "p_like_raw": (
                    None
                    if "p_like_raw" not in row
                    else float(row["p_like_raw"])
                ),
            }
        )

    all_results = []
    for room_id in sorted(grouped):
        items = grouped[room_id]
        items.sort(key=lambda item: item["candidate_index"])
        raw_scores = [item["reward_score"] for item in items]
        z_scores = normalize(raw_scores)
        order = sorted(
            range(len(items)),
            key=lambda index: (-raw_scores[index], index),
        )

        reference_path = room_assets[room_id]["reference_path"]
        reference = Image.open(reference_path).convert("RGB")
        sorted_images = []
        labels = []
        ranked_records = []

        for rank, item_index in enumerate(order, start=1):
            item = items[item_index]
            candidate_path = Path(item["candidate_path"])
            sorted_images.append(Image.open(candidate_path).convert("RGB"))
            labels.append(
                f"rank={rank} reward={item['reward_score']:.4f} "
                f"z={z_scores[item_index]:+.3f} file={candidate_path.name}"
            )
            ranked_records.append(
                {
                    "rank": rank,
                    "candidate_index": item["candidate_index"],
                    "candidate_path": str(candidate_path),
                    "reward_score": item["reward_score"],
                    "normalized_reward": float(z_scores[item_index]),
                    "logit_A_like": item["logit_A_like"],
                    "logit_B_dislike": item["logit_B_dislike"],
                    "p_like_raw": item["p_like_raw"],
                }
            )

        top1_source = Path(ranked_records[0]["candidate_path"])
        top1_path = top1_dir / f"{room_id}_top1.png"
        shutil.copy2(top1_source, top1_path)
        sheet_path = sheets_dir / f"{room_id}.jpg"
        save_contact_sheet(
            reference,
            sorted_images,
            labels,
            sheet_path,
        )

        result = {
            "room_id": room_id,
            "reference": str(reference_path),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_epoch": checkpoint.epoch,
            "top1_source": str(top1_source),
            "top1_copy": str(top1_path),
            "contact_sheet": str(sheet_path),
            "reward_range": float(max(raw_scores) - min(raw_scores)),
            "top1_top2_margin": (
                float(raw_scores[order[0]] - raw_scores[order[1]])
                if len(order) >= 2
                else None
            ),
            "candidates": ranked_records,
        }
        write_json(output_dir / "rooms" / room_id / "p1_result.json", result)
        all_results.append(result)
        LOGGER.info(
            "[%s] top1=%s reward=%.4f margin12=%s",
            room_id,
            top1_source.name,
            ranked_records[0]["reward_score"],
            result["top1_top2_margin"],
        )

    rankings_path = output_dir / "p1_rankings.jsonl"
    with rankings_path.open("w", encoding="utf-8") as file:
        for result in all_results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "status": "complete",
        "rooms_scored": len(all_results),
        "candidates_scored": sum(len(item["candidates"]) for item in all_results),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_epoch": checkpoint.epoch,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "rankings_jsonl": str(rankings_path),
    }
    write_json(output_dir / "summary.json", summary)
    LOGGER.info("Finished: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
