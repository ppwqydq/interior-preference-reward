#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 Qwen P2 Scalar Reward Head，对 Klein 已生成候选做离线组内重排。

设计目标：
- 不修改现有 P1 排序脚本；
- 复用 pairwise_checkpoint.py 正式加载 P2 LoRA 与 reward_head.pt；
- 复用 RewardScoringSample / QwenABRewardBackend 的双图输入；
- 直接从 Qwen backbone 最后一个有效 token hidden state 计算 scalar reward；
- 输出结构与 P1 基本一致，便于比较。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger("rerank_klein_candidates_p2")


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


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def find_pairwise_checkpoint_loader(module: Any) -> Callable[..., Any]:
    """优先使用约定名称；仓库版本不同则按签名安全发现。"""
    preferred_names = (
        "load_pairwise_checkpoint",
        "load_pairwise_checkpoint_config",
        "read_pairwise_checkpoint",
    )
    for name in preferred_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn

    candidates: list[tuple[str, Callable[..., Any]]] = []
    for name in dir(module):
        lower = name.lower()
        fn = getattr(module, name)
        if not callable(fn):
            continue
        if "checkpoint" not in lower or "load" not in lower:
            continue
        if "backend" in lower:
            continue
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        candidates.append((name, fn))

    if len(candidates) == 1:
        LOGGER.info("自动发现 Pairwise Checkpoint loader：%s", candidates[0][0])
        return candidates[0][1]

    names = ", ".join(name for name, _ in candidates) or "无"
    raise RuntimeError(
        "无法唯一确定 Pairwise Checkpoint loader。"
        f"候选：{names}"
    )


def call_checkpoint_loader(loader: Callable[..., Any], checkpoint_dir: Path) -> Any:
    signature = inspect.signature(loader)
    available = {
        "checkpoint_dir": checkpoint_dir,
        "path": checkpoint_dir,
        "directory": checkpoint_dir,
    }
    kwargs = {
        name: value
        for name, value in available.items()
        if name in signature.parameters
    }
    if kwargs:
        return loader(**kwargs)

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if positional:
        return loader(checkpoint_dir)

    raise TypeError(f"无法调用 Checkpoint loader，签名：{signature}")


def move_inputs_to_device(inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def resolve_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """取得带 LoRA 的 Qwen backbone，避免请求全部层 hidden_states。"""
    full_model = model
    if hasattr(model, "get_base_model"):
        try:
            full_model = model.get_base_model()
        except Exception:
            full_model = model

    backbone = getattr(full_model, "model", None)
    if isinstance(backbone, torch.nn.Module):
        return backbone

    # 一些 PEFT 包装结构需要再向下取一层。
    base_model = getattr(full_model, "base_model", None)
    nested_model = getattr(base_model, "model", None)
    nested_backbone = getattr(nested_model, "model", None)
    if isinstance(nested_backbone, torch.nn.Module):
        return nested_backbone

    raise RuntimeError(
        "无法从 Backend 模型定位 Qwen backbone；"
        f"model_type={type(model)!r}"
    )


def filter_forward_kwargs(module: torch.nn.Module, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """仅传入 backbone.forward 接受的参数，兼容不同 Transformers 版本。"""
    signature = inspect.signature(module.forward)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    if accepts_kwargs:
        return dict(inputs)
    return {
        key: value
        for key, value in inputs.items()
        if key in signature.parameters
    }


def last_valid_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask 应为二维张量，"
            f"实际 shape={tuple(attention_mask.shape)}"
        )
    mask = attention_mask.to(dtype=torch.long)
    positions = torch.arange(
        mask.shape[1],
        device=mask.device,
        dtype=torch.long,
    ).unsqueeze(0)
    weighted = torch.where(mask > 0, positions, torch.full_like(positions, -1))
    last = weighted.max(dim=1).values
    if torch.any(last < 0):
        raise ValueError("存在 attention_mask 全为 0 的样本")
    return last


@torch.inference_mode()
def score_scalar_samples(
    *,
    backend: Any,
    reward_head: torch.nn.Module,
    samples: Sequence[Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    if not samples:
        raise ValueError("评分样本不能为空")

    backend.model.eval()
    reward_head.eval()
    device = torch.device(backend.device)
    backbone = resolve_backbone(backend.model)
    LOGGER.info(
        "Scalar scoring backbone=%s reward_head=%s device=%s",
        type(backbone).__name__,
        type(reward_head).__name__,
        device,
    )

    results: list[dict[str, Any]] = []

    for batch_samples in batched(samples, batch_size):
        inputs = backend.make_inputs(batch_samples)
        if not isinstance(inputs, Mapping):
            # transformers.BatchFeature 也实现 Mapping；若不是则尽早报清晰错误。
            raise TypeError(
                "backend.make_inputs 返回值不是 Mapping："
                f"{type(inputs)!r}"
            )
        model_inputs = move_inputs_to_device(inputs, device)
        if "attention_mask" not in model_inputs:
            raise KeyError("Backend 输入缺少 attention_mask")

        forward_kwargs = filter_forward_kwargs(backbone, model_inputs)
        forward_signature = inspect.signature(backbone.forward)
        if "return_dict" in forward_signature.parameters:
            forward_kwargs["return_dict"] = True
        if "use_cache" in forward_signature.parameters:
            forward_kwargs["use_cache"] = False

        outputs = backbone(**forward_kwargs)
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            if isinstance(outputs, (tuple, list)) and outputs:
                hidden_states = outputs[0]
            else:
                raise RuntimeError(
                    "Qwen backbone 输出中没有 last_hidden_state"
                )

        last_positions = last_valid_positions(model_inputs["attention_mask"])
        batch_indices = torch.arange(
            hidden_states.shape[0],
            device=hidden_states.device,
        )
        last_hidden = hidden_states[batch_indices, last_positions]

        head_device = next(reward_head.parameters()).device
        head_dtype = next(reward_head.parameters()).dtype
        last_hidden = last_hidden.to(device=head_device, dtype=head_dtype)
        scores = reward_head(last_hidden)
        if scores.ndim == 2 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        if scores.ndim != 1:
            raise RuntimeError(
                "Scalar Reward Head 输出应为 [B] 或 [B,1]，"
                f"实际 shape={tuple(scores.shape)}"
            )

        scores_cpu = scores.detach().float().cpu().tolist()
        positions_cpu = last_positions.detach().cpu().tolist()
        for sample, score, position in zip(
            batch_samples,
            scores_cpu,
            positions_cpu,
        ):
            results.append(
                {
                    "sample_id": sample.sample_id,
                    "reward_score": float(score),
                    "last_token_position": int(position),
                }
            )

        del outputs, hidden_states, last_hidden, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qwen_project",
        default="/root/qwen_pref_reward",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=(
            "/root/autodl-tmp/qwen_pref_reward_outputs/"
            "qwen3_vl_8b_layout100_pairwise_p2_scalar_512/epoch_23"
        ),
    )
    parser.add_argument(
        "--source_dir",
        default=(
            "/root/autodl-tmp/klein_generation_ranker_outputs/"
            "dev_livingroom_20x6"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn_implementation", default="")
    parser.add_argument("--limit_rooms", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有 p2_reranked 输出目录",
    )
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
    from preference_reward.inference.scoring import RewardScoringSample

    pairwise_module = importlib.import_module(
        "preference_reward.inference.pairwise_checkpoint"
    )
    checkpoint_loader = find_pairwise_checkpoint_loader(pairwise_module)
    load_pairwise_backend = getattr(
        pairwise_module,
        "load_pairwise_backend",
    )

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint = call_checkpoint_loader(checkpoint_loader, checkpoint_dir)

    score_type = str(getattr(checkpoint, "score_type", ""))
    if score_type and score_type != "scalar_head":
        raise ValueError(
            "该脚本只支持 P2 scalar_head，"
            f"当前 score_type={score_type!r}"
        )

    device = torch.device(args.device)
    backend, reward_head = load_pairwise_backend(
        checkpoint=checkpoint,
        device=device,
        logger=LOGGER,
        attn_implementation=args.attn_implementation,
    )
    if reward_head is None:
        raise RuntimeError(
            "P2 Checkpoint 未加载出 Scalar Reward Head；"
            "请确认 reward_head.pt 存在且 score_type=scalar_head"
        )

    source_dir = Path(args.source_dir).expanduser().resolve()
    rooms_root = source_dir / "rooms"
    output_dir = source_dir / "p2_reranked"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"输出目录已存在：{output_dir}\n"
                "如需覆盖，请添加 --overwrite"
            )
        shutil.rmtree(output_dir)

    sheets_dir = output_dir / "contact_sheets"
    top1_dir = output_dir / "top1"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    top1_dir.mkdir(parents=True, exist_ok=True)

    if not rooms_root.is_dir():
        raise FileNotFoundError(f"没有找到 rooms 目录：{rooms_root}")
    room_dirs = sorted(path for path in rooms_root.iterdir() if path.is_dir())
    if args.limit_rooms > 0:
        room_dirs = room_dirs[: args.limit_rooms]
    if not room_dirs:
        raise FileNotFoundError(f"没有找到房间目录：{rooms_root}")

    scoring_samples: list[Any] = []
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
        "准备 P2 评分：rooms=%d candidates=%d checkpoint=%s epoch=%s score_type=%s",
        len(room_assets),
        len(scoring_samples),
        checkpoint_dir,
        getattr(checkpoint, "epoch", None),
        getattr(checkpoint, "score_type", None),
    )

    rows = score_scalar_samples(
        backend=backend,
        reward_head=reward_head,
        samples=scoring_samples,
        batch_size=args.batch_size,
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
                "last_token_position": int(row["last_token_position"]),
            }
        )

    all_results: list[dict[str, Any]] = []
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
        sorted_images: list[Image.Image] = []
        labels: list[str] = []
        ranked_records: list[dict[str, Any]] = []

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
                    "last_token_position": item["last_token_position"],
                }
            )

        top1_source = Path(ranked_records[0]["candidate_path"])
        top1_path = top1_dir / f"{room_id}_top1.png"
        shutil.copy2(top1_source, top1_path)
        sheet_path = sheets_dir / f"{room_id}.jpg"
        save_contact_sheet(reference, sorted_images, labels, sheet_path)

        result = {
            "room_id": room_id,
            "reference": str(reference_path),
            "score_type": "scalar_head",
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_epoch": getattr(checkpoint, "epoch", None),
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
        write_json(output_dir / "rooms" / room_id / "p2_result.json", result)
        all_results.append(result)
        LOGGER.info(
            "[%s] P2 top1=%s reward=%.4f margin12=%s",
            room_id,
            top1_source.name,
            ranked_records[0]["reward_score"],
            result["top1_top2_margin"],
        )

    rankings_path = output_dir / "p2_rankings.jsonl"
    with rankings_path.open("w", encoding="utf-8") as file:
        for result in all_results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "status": "complete",
        "score_type": "scalar_head",
        "rooms_scored": len(all_results),
        "candidates_scored": sum(len(item["candidates"]) for item in all_results),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_epoch": getattr(checkpoint, "epoch", None),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "rankings_jsonl": str(rankings_path),
    }
    write_json(output_dir / "summary.json", summary)
    LOGGER.info("Finished: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
