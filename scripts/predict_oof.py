#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""依次预测各折 Outer Holdout，并合并完整 OOF 结果。"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


from preference_reward.common.config import (
    load_yaml_config,
    resolve_project_path,
)
from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)
from preference_reward.common.logging_utils import (
    setup_logger,
)
from preference_reward.data.manifest import (
    read_preference_manifest,
)
from preference_reward.evaluation.oof import (
    build_confidence_report,
    compute_oof_metrics,
    enrich_oof_predictions,
    validate_oof_coverage,
)
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


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "加载每折最佳 Checkpoint，预测对应 Outer Holdout，"
            "并合并完整 OOF 预测。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "qwen8b_layout_512.yaml"
        ),
    )
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "oof_4fold"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "qwen3_vl_8b_layout_ab_512_oof"
        ),
    )
    parser.add_argument(
        "--start_fold",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--end_fold",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新计算已经存在且校验通过的单折预测。",
    )
    parser.add_argument(
        "--skip_image_path_check",
        action="store_true",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    """读取 JSON 对象。"""

    if not path.is_file():
        raise FileNotFoundError(
            f"JSON 文件不存在：{path}"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise TypeError(
            f"JSON 根节点必须是对象：{path}"
        )

    return data


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL。"""

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


def prepare_fold_predictions(
    fold: int,
    split_dir: Path,
    output_root: Path,
    config: Mapping[str, Any],
    batch_size: int,
    validate_image_paths: bool,
    logger: Any,
) -> List[Dict[str, Any]]:
    """加载单折最佳模型并预测 Outer Holdout。"""

    fold_split_dir = (
        split_dir / f"fold_{fold}"
    )
    fold_output_dir = (
        output_root / f"fold_{fold}"
    )
    outer_manifest = (
        fold_split_dir
        / "outer_holdout.jsonl"
    )
    best_pointer_path = (
        fold_output_dir
        / "best_checkpoint.json"
    )

    best_pointer = read_json(
        best_pointer_path
    )

    adapter_path = Path(
        str(best_pointer["adapter_path"])
    ).expanduser().resolve()

    checkpoint = (
        load_reward_checkpoint_config(
            checkpoint_dir=adapter_path,
            validate_paths=True,
        )
    )

    logger.info(
        "Fold %d | best_epoch=%s，"
        "best_value=%s，adapter=%s",
        fold,
        best_pointer.get("epoch"),
        best_pointer.get("value"),
        adapter_path,
    )

    samples = read_preference_manifest(
        manifest_path=outer_manifest,
        project_root=PROJECT_ROOT,
        validate_image_paths=(
            validate_image_paths
        ),
    )

    scoring_samples = [
        RewardScoringSample(
            sample_id=sample.sample_id,
            empty_room_image=(
                sample.empty_room_image
            ),
            generated_furniture_image=(
                sample.generated_furniture_image
            ),
            metadata={
                "fold": fold,
                "label": sample.label,
            },
        )
        for sample in samples
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        logger.warning(
            "CUDA 不可用，8B 模型推理将非常慢"
        )

    backend = load_qwen_ab_backend(
        checkpoint=checkpoint,
        device=device,
        logger=logger,
        attn_implementation=str(
            config["model"].get(
                "attn_implementation",
                "",
            )
        ),
    )

    try:
        raw_results = score_reward_samples(
            backend=backend,
            samples=scoring_samples,
            batch_size=batch_size,
            negative_weight=(
                checkpoint.negative_weight
            ),
        )
    finally:
        del backend
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sample_by_id = {
        sample.sample_id: sample
        for sample in samples
    }

    predictions: List[Dict[str, Any]] = []

    for result in raw_results:
        sample_id = str(
            result["sample_id"]
        )
        sample = sample_by_id[sample_id]

        predictions.append(
            {
                "sample_id": sample_id,
                "fold": fold,
                "label": sample.label,
                "empty_room_image": str(
                    sample.empty_room_image
                ),
                "generated_furniture_image": str(
                    sample
                    .generated_furniture_image
                ),
                "logit_A_like": float(
                    result["logit_A_like"]
                ),
                "logit_B_dislike": float(
                    result["logit_B_dislike"]
                ),
                "reward_score": float(
                    result["reward_score"]
                ),
                "p_like_raw": float(
                    result["p_like_raw"]
                ),
                "p_like_prior_corrected": float(
                    result[
                        "p_like_prior_corrected"
                    ]
                ),
                "last_valid_position": int(
                    result[
                        "last_valid_position"
                    ]
                ),
                "checkpoint_epoch": (
                    checkpoint.epoch
                ),
                "checkpoint_dir": str(
                    checkpoint.checkpoint_dir
                ),
                "negative_weight": float(
                    checkpoint.negative_weight
                ),
            }
        )

    expected_ids = {
        sample.sample_id
        for sample in samples
    }
    actual_ids = {
        str(row["sample_id"])
        for row in predictions
    }

    if actual_ids != expected_ids:
        missing = sorted(
            expected_ids - actual_ids
        )
        unexpected = sorted(
            actual_ids - expected_ids
        )
        raise RuntimeError(
            f"Fold {fold} 预测覆盖失败："
            f"missing={missing[:10]}，"
            f"unexpected={unexpected[:10]}"
        )

    return predictions


def main() -> None:
    """执行完整 OOF 推理流水线。"""

    args = parse_args()

    if args.start_fold <= 0:
        raise ValueError(
            "start_fold 必须大于 0"
        )

    if args.end_fold < args.start_fold:
        raise ValueError(
            "end_fold 不能小于 start_fold"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size 必须大于 0"
        )

    config_path = args.config.resolve()
    split_dir = args.split_dir.resolve()
    output_root = args.output_root.resolve()

    config = load_yaml_config(
        config_path
    )

    threshold = float(
        config["evaluation"]["threshold"]
    )
    ece_bins = int(
        config["evaluation"]["ece_bins"]
    )

    logger, log_path = setup_logger(
        name="oof_prediction",
        log_dir=resolve_project_path(
            PROJECT_ROOT,
            config["paths"]["log_dir"],
        ),
        prefix="oof_prediction",
    )

    logger.info(
        "配置文件：%s",
        config_path,
    )
    logger.info(
        "划分目录：%s",
        split_dir,
    )
    logger.info(
        "训练输出根目录：%s",
        output_root,
    )
    logger.info(
        "日志文件：%s",
        log_path,
    )

    assignments_path = (
        split_dir
        / "oof_assignments.jsonl"
    )
    assignments = read_jsonl(
        assignments_path
    )

    status_path = (
        output_root
        / "oof_prediction_status.json"
    )
    merged_output_path = (
        output_root
        / "oof_predictions.jsonl"
    )
    metrics_output_path = (
        output_root
        / "oof_metrics.json"
    )
    confidence_output_path = (
        output_root
        / "oof_confidence_report.json"
    )

    status: Dict[str, Any] = {
        "status": "running",
        "started_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "config": str(config_path),
        "split_dir": str(split_dir),
        "output_root": str(output_root),
        "start_fold": args.start_fold,
        "end_fold": args.end_fold,
        "batch_size": args.batch_size,
        "folds": {},
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    write_json_atomic(
        status,
        status_path,
    )

    all_predictions: List[
        Dict[str, Any]
    ] = []

    try:
        for fold in range(
            args.start_fold,
            args.end_fold + 1,
        ):
            fold_output_dir = (
                output_root / f"fold_{fold}"
            )
            fold_prediction_path = (
                fold_output_dir
                / "outer_holdout_predictions.jsonl"
            )
            fold_metrics_path = (
                fold_output_dir
                / "outer_holdout_metrics.json"
            )

            status["folds"][str(fold)] = {
                "status": "running",
                "started_at": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
            }
            write_json_atomic(
                status,
                status_path,
            )

            if args.dry_run:
                logger.info(
                    "DRY RUN | Fold %d | "
                    "split=%s | output=%s",
                    fold,
                    (
                        split_dir
                        / f"fold_{fold}"
                        / "outer_holdout.jsonl"
                    ),
                    fold_prediction_path,
                )
                status["folds"][str(fold)][
                    "status"
                ] = "dry_run"
                continue

            predictions: List[
                Dict[str, Any]
            ]

            if (
                fold_prediction_path.is_file()
                and not args.force
            ):
                logger.info(
                    "Fold %d 已存在预测，"
                    "读取并校验：%s",
                    fold,
                    fold_prediction_path,
                )
                predictions = read_jsonl(
                    fold_prediction_path
                )
            else:
                predictions = (
                    prepare_fold_predictions(
                        fold=fold,
                        split_dir=split_dir,
                        output_root=output_root,
                        config=config,
                        batch_size=(
                            args.batch_size
                        ),
                        validate_image_paths=(
                            not args
                            .skip_image_path_check
                        ),
                        logger=logger,
                    )
                )

            enriched = enrich_oof_predictions(
                predictions=predictions,
                threshold=threshold,
            )

            expected_fold_assignments = [
                row
                for row in assignments
                if int(row["outer_fold"])
                == fold
            ]

            validate_oof_coverage(
                predictions=enriched,
                assignments=(
                    expected_fold_assignments
                ),
            )

            fold_metrics = compute_oof_metrics(
                predictions=enriched,
                threshold=threshold,
                ece_bins=ece_bins,
            )

            write_jsonl_atomic(
                enriched,
                fold_prediction_path,
            )
            write_json_atomic(
                {
                    "fold": fold,
                    "predictions_path": str(
                        fold_prediction_path
                    ),
                    "metrics": fold_metrics,
                },
                fold_metrics_path,
            )

            all_predictions.extend(
                enriched
            )

            status["folds"][str(fold)].update(
                {
                    "status": "completed",
                    "finished_at": (
                        datetime.now().isoformat(
                            timespec="seconds"
                        )
                    ),
                    "samples": len(enriched),
                    "roc_auc": (
                        fold_metrics["roc_auc"]
                    ),
                    "predictions_path": str(
                        fold_prediction_path
                    ),
                    "metrics_path": str(
                        fold_metrics_path
                    ),
                }
            )
            write_json_atomic(
                status,
                status_path,
            )

            logger.info(
                "Fold %d 完成：samples=%d，"
                "roc_auc=%s",
                fold,
                len(enriched),
                fold_metrics["roc_auc"],
            )

        if args.dry_run:
            status["status"] = "dry_run"
            status["finished_at"] = (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            )
            write_json_atomic(
                status,
                status_path,
            )
            return

        all_assignment_folds = sorted(
            {
                int(row["outer_fold"])
                for row in assignments
            }
        )
        requested_folds = list(
            range(
                args.start_fold,
                args.end_fold + 1,
            )
        )

        if requested_folds != all_assignment_folds:
            status["status"] = "partial_completed"
            status["finished_at"] = (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            )
            status["note"] = (
                "本次只处理部分 Fold，"
                "未生成全量合并 OOF 文件。"
                "使用完整 Fold 范围运行后才会生成。"
            )
            write_json_atomic(
                status,
                status_path,
            )
            logger.info(
                "部分 Fold 推理完成；"
                "未生成全量 OOF 合并文件。"
            )
            return

        enriched_all = enrich_oof_predictions(
            predictions=all_predictions,
            threshold=threshold,
        )

        coverage = validate_oof_coverage(
            predictions=enriched_all,
            assignments=assignments,
        )
        metrics = compute_oof_metrics(
            predictions=enriched_all,
            threshold=threshold,
            ece_bins=ece_bins,
        )
        confidence_report = (
            build_confidence_report(
                predictions=enriched_all,
                top_k=args.top_k,
            )
        )

        write_jsonl_atomic(
            enriched_all,
            merged_output_path,
        )
        write_json_atomic(
            {
                "coverage": coverage,
                "metrics": metrics,
                "warning": (
                    "合并 OOF 指标来自四个独立模型。"
                    "不同 Fold 的分数尺度可能存在偏移，"
                    "因此整体 ROC-AUC 仅用于诊断；"
                    "样本可信度也不应视为真实标签正确概率。"
                ),
            },
            metrics_output_path,
        )
        write_json_atomic(
            confidence_report,
            confidence_output_path,
        )

        status.update(
            {
                "status": "completed",
                "finished_at": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
                "coverage": coverage,
                "metrics_path": str(
                    metrics_output_path
                ),
                "predictions_path": str(
                    merged_output_path
                ),
                "confidence_report_path": str(
                    confidence_output_path
                ),
            }
        )
        write_json_atomic(
            status,
            status_path,
        )

        logger.info(
            "OOF 推理完成：samples=%d，"
            "roc_auc=%s",
            coverage["predicted_samples"],
            metrics["roc_auc"],
        )
        logger.info(
            "预测文件：%s",
            merged_output_path,
        )
        logger.info(
            "可信度报告：%s",
            confidence_output_path,
        )

    except Exception as exc:
        status["status"] = "failed"
        status["finished_at"] = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )
        status["error"] = repr(exc)
        write_json_atomic(
            status,
            status_path,
        )
        raise


if __name__ == "__main__":
    main()
