#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""通用双图 Pairwise Reward Model 评估入口。

流程：

    Checkpoint
    → 独立加载推理模型
    → 读取 Pairwise Manifest
    → 展开 positive / negative 候选
    → 批量计算 reward score
    → 汇总 Pairwise 指标
    → 原子保存评估结果

本脚本不包含训练逻辑，也不修改模型参数。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


import torch

from preference_reward.common.config import (
    resolve_project_path,
)
from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)
from preference_reward.common.logging_utils import (
    setup_logger,
)
from preference_reward.data.pairwise_manifest import (
    read_pairwise_manifest,
)
from preference_reward.evaluation.pairwise import (
    build_pairwise_scoring_samples,
    evaluate_pairwise_scores,
)
from preference_reward.inference.checkpoint import (
    load_reward_checkpoint_config,
)
from preference_reward.inference.qwen_ab_loader import (
    load_qwen_ab_backend,
)
from preference_reward.inference.scoring import (
    score_reward_samples,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "评估双图 Reward Model 的 "
            "positive / negative Pairwise 排序能力。"
        )
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help=(
            "包含 checkpoint_config.json 和 "
            "LoRA Adapter 的 Epoch 目录。"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Pairwise JSONL 清单路径。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="评估结果输出目录。",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="日志输出目录。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="单候选推理 Batch Size。",
    )
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=5000,
        help="分组 Bootstrap 迭代次数。",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=42,
        help="Bootstrap 随机种子。",
    )
    parser.add_argument(
        "--tie_tolerance",
        type=float,
        default=1e-8,
        help=(
            "绝对 Pairwise Margin 小于等于该值时"
            "视为平局。"
        ),
    )
    parser.add_argument(
        "--difficult_count",
        type=int,
        default=10,
        help="保存绝对 Margin 最小的困难 Pair 数量。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备，例如 cuda、cuda:0 或 cpu。",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="",
        help=(
            "可选 Transformers Attention 实现；"
            "留空时使用模型默认值。"
        ),
    )
    parser.add_argument(
        "--skip_image_path_check",
        action="store_true",
        help="跳过 Pairwise Manifest 图片路径检查。",
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """校验命令行参数。"""

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size 必须大于 0"
        )

    if args.bootstrap_iterations <= 0:
        raise ValueError(
            "bootstrap_iterations 必须大于 0"
        )

    if args.tie_tolerance < 0:
        raise ValueError(
            "tie_tolerance 不能小于 0"
        )

    if args.difficult_count < 0:
        raise ValueError(
            "difficult_count 不能小于 0"
        )


def resolve_device(
    value: str,
) -> torch.device:
    """解析并校验推理设备。"""

    device = torch.device(value)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "指定了 CUDA 推理，但当前 CUDA 不可用"
        )

    return device


def build_run_config(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    log_path: Path,
    device: torch.device,
    checkpoint_epoch: int | None,
    num_pairs: int,
    num_scoring_samples: int,
) -> Dict[str, Any]:
    """构造可复现的评估运行配置。"""

    return {
        "created_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "project_root": str(PROJECT_ROOT),
        "checkpoint_dir": str(
            checkpoint_dir
        ),
        "checkpoint_epoch": (
            checkpoint_epoch
        ),
        "manifest_path": str(
            manifest_path
        ),
        "output_dir": str(
            output_dir
        ),
        "log_path": str(
            log_path
        ),
        "device": str(device),
        "batch_size": int(
            args.batch_size
        ),
        "bootstrap_iterations": int(
            args.bootstrap_iterations
        ),
        "bootstrap_seed": int(
            args.bootstrap_seed
        ),
        "tie_tolerance": float(
            args.tie_tolerance
        ),
        "difficult_count": int(
            args.difficult_count
        ),
        "attn_implementation": str(
            args.attn_implementation
        ),
        "validate_image_paths": bool(
            not args.skip_image_path_check
        ),
        "num_pairs": int(
            num_pairs
        ),
        "num_scoring_samples": int(
            num_scoring_samples
        ),
    }


def log_pairwise_summary(
    logger: Any,
    metrics: Dict[str, Any],
) -> None:
    """输出主要 Pairwise 指标。"""

    logger.info(
        "Pair 数量：%d",
        metrics["num_pairs"],
    )
    logger.info(
        "正确=%d，错误=%d，平局=%d",
        metrics["num_correct"],
        metrics["num_incorrect"],
        metrics["num_ties"],
    )
    logger.info(
        "Pairwise Accuracy：%.6f",
        metrics[
            "pairwise_accuracy_strict"
        ],
    )
    logger.info(
        "Pairwise Accuracy 95%% CI："
        "[%.6f, %.6f]",
        metrics[
            "pairwise_accuracy_ci95_lower"
        ],
        metrics[
            "pairwise_accuracy_ci95_upper"
        ],
    )
    logger.info(
        "平均 GOOD-BAD Margin：%.6f",
        metrics[
            "mean_pairwise_margin"
        ],
    )
    logger.info(
        "中位数 GOOD-BAD Margin：%.6f",
        metrics[
            "median_pairwise_margin"
        ],
    )
    logger.info(
        "Pointwise ROC-AUC：%s",
        (
            f'{metrics["pointwise_roc_auc"]:.6f}'
            if (
                metrics["pointwise_roc_auc"]
                is not None
            )
            else "None"
        ),
    )
    logger.info(
        "唯一内容锚点：%d",
        metrics["num_unique_anchors"],
    )


def main() -> None:
    """执行完整 Pairwise 评估。"""

    args = parse_args()
    validate_args(args)

    checkpoint_dir = resolve_project_path(
        PROJECT_ROOT,
        args.checkpoint_dir,
    )
    manifest_path = resolve_project_path(
        PROJECT_ROOT,
        args.manifest,
    )
    output_dir = resolve_project_path(
        PROJECT_ROOT,
        args.output_dir,
    )
    log_dir = resolve_project_path(
        PROJECT_ROOT,
        args.log_dir,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_name = (
        checkpoint_dir.parent.name
        + "_"
        + checkpoint_dir.name
    )

    logger, log_path = setup_logger(
        name="qwen_pairwise_evaluation",
        log_dir=log_dir,
        prefix=(
            f"pairwise_{checkpoint_name}"
        ),
    )

    logger.info(
        "项目根目录：%s",
        PROJECT_ROOT,
    )
    logger.info(
        "Checkpoint：%s",
        checkpoint_dir,
    )
    logger.info(
        "Pairwise Manifest：%s",
        manifest_path,
    )
    logger.info(
        "输出目录：%s",
        output_dir,
    )
    logger.info(
        "日志文件：%s",
        log_path,
    )

    checkpoint = (
        load_reward_checkpoint_config(
            checkpoint_dir
        )
    )

    logger.info(
        "Checkpoint epoch：%s",
        checkpoint.epoch,
    )
    logger.info(
        "基础模型：%s",
        checkpoint.base_model_path,
    )
    logger.info(
        "Processor：%s",
        checkpoint.processor_path,
    )
    logger.info(
        "Adapter：%s",
        checkpoint.adapter_path,
    )
    logger.info(
        "max_pixels：%d",
        checkpoint.max_pixels,
    )
    logger.info(
        "negative_weight：%.10f",
        checkpoint.negative_weight,
    )

    pairwise_samples = (
        read_pairwise_manifest(
            manifest_path=manifest_path,
            project_root=PROJECT_ROOT,
            validate_image_paths=(
                not args.skip_image_path_check
            ),
        )
    )

    scoring_samples = (
        build_pairwise_scoring_samples(
            pairwise_samples
        )
    )

    logger.info(
        "Pair 数量：%d",
        len(pairwise_samples),
    )
    logger.info(
        "待评分候选数量：%d",
        len(scoring_samples),
    )

    device = resolve_device(
        args.device
    )

    logger.info(
        "推理设备：%s",
        device,
    )

    backend = load_qwen_ab_backend(
        checkpoint=checkpoint,
        device=device,
        logger=logger,
        attn_implementation=(
            args.attn_implementation
        ),
    )

    logger.info(
        "开始计算候选 Reward 分数"
    )

    scoring_rows = score_reward_samples(
        backend=backend,
        samples=scoring_samples,
        batch_size=args.batch_size,
        negative_weight=(
            checkpoint.negative_weight
        ),
    )

    if (
        len(scoring_rows)
        != len(scoring_samples)
    ):
        raise RuntimeError(
            "评分结果数量不一致："
            f"输入={len(scoring_samples)}，"
            f"输出={len(scoring_rows)}"
        )

    logger.info(
        "候选评分完成：%d 条",
        len(scoring_rows),
    )

    metrics, pair_rows = (
        evaluate_pairwise_scores(
            pairwise_samples=(
                pairwise_samples
            ),
            scoring_rows=scoring_rows,
            tie_tolerance=(
                args.tie_tolerance
            ),
            bootstrap_iterations=(
                args.bootstrap_iterations
            ),
            bootstrap_seed=(
                args.bootstrap_seed
            ),
        )
    )

    error_rows = sorted(
        [
            row
            for row in pair_rows
            if row["is_incorrect"]
        ],
        key=lambda row: float(
            row["pairwise_margin"]
        ),
    )

    tie_rows = sorted(
        [
            row
            for row in pair_rows
            if row["is_tie"]
        ],
        key=lambda row: abs(
            float(
                row["pairwise_margin"]
            )
        ),
    )

    difficult_rows = sorted(
        pair_rows,
        key=lambda row: abs(
            float(
                row["pairwise_margin"]
            )
        ),
    )[:args.difficult_count]

    run_config = build_run_config(
        args=args,
        checkpoint_dir=checkpoint_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
        log_path=log_path,
        device=device,
        checkpoint_epoch=(
            checkpoint.epoch
        ),
        num_pairs=len(
            pairwise_samples
        ),
        num_scoring_samples=len(
            scoring_samples
        ),
    )

    metrics_output = {
        **metrics,
        "checkpoint_epoch": (
            checkpoint.epoch
        ),
        "checkpoint_dir": str(
            checkpoint_dir
        ),
        "manifest_path": str(
            manifest_path
        ),
        "num_candidate_scores": int(
            len(scoring_rows)
        ),
    }

    write_json_atomic(
        run_config,
        output_dir
        / "evaluation_config.json",
    )
    write_json_atomic(
        metrics_output,
        output_dir
        / "pairwise_metrics.json",
    )
    write_jsonl_atomic(
        scoring_rows,
        output_dir
        / "candidate_scores.jsonl",
    )
    write_jsonl_atomic(
        pair_rows,
        output_dir
        / "pairwise_predictions.jsonl",
    )
    write_jsonl_atomic(
        error_rows,
        output_dir
        / "pairwise_errors.jsonl",
    )
    write_jsonl_atomic(
        tie_rows,
        output_dir
        / "pairwise_ties.jsonl",
    )
    write_jsonl_atomic(
        difficult_rows,
        output_dir
        / "pairwise_difficult.jsonl",
    )

    log_pairwise_summary(
        logger,
        metrics,
    )

    if error_rows:
        logger.info(
            "最严重的错误 Pair："
        )

        for row in error_rows[:5]:
            logger.info(
                "pair_id=%s "
                "GOOD=%.6f "
                "BAD=%.6f "
                "margin=%.6f",
                row["pair_id"],
                row[
                    "positive_reward_score"
                ],
                row[
                    "negative_reward_score"
                ],
                row["pairwise_margin"],
            )

    logger.info(
        "评估完成，结果目录：%s",
        output_dir,
    )


if __name__ == "__main__":
    main()
