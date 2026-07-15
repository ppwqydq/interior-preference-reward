#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""使用固定 Checkpoint 评估 Pointwise 训练清单。

该脚本只负责：

1. 读取最佳 Checkpoint；
2. 加载冻结的推理 Backend；
3. 对指定 Pointwise Manifest 运行分类评估；
4. 保存指标和逐样本预测。

不会修改模型、训练器或数据。
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)
from preference_reward.data.manifest import (
    count_labels,
    read_preference_manifest,
)
from preference_reward.evaluation.classification import (
    evaluate_samples,
)
from preference_reward.inference.checkpoint import (
    load_reward_checkpoint_config,
)
from preference_reward.inference.qwen_ab_loader import (
    load_qwen_ab_backend,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="评估固定 Checkpoint 在 Pointwise 清单上的表现。"
    )

    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="包含 best_checkpoint.json 的训练输出目录。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="需要评估的 Pointwise JSONL 清单。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ece_bins",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="",
    )
    parser.add_argument(
        "--skip_image_path_check",
        action="store_true",
    )

    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """把相对路径转换为项目绝对路径。"""

    expanded = path.expanduser()

    if expanded.is_absolute():
        return expanded.resolve()

    return (PROJECT_ROOT / expanded).resolve()


def read_best_checkpoint(
    output_root: Path,
) -> tuple[int, Path]:
    """从 best_checkpoint.json 读取最佳 Epoch 和目录。"""

    best_path = output_root / "best_checkpoint.json"

    if not best_path.is_file():
        raise FileNotFoundError(
            f"缺少最佳 Checkpoint 记录：{best_path}"
        )

    record = json.loads(
        best_path.read_text(encoding="utf-8")
    )

    epoch = int(record["epoch"])

    adapter_value = record.get("adapter_path")

    if adapter_value:
        checkpoint_dir = Path(
            str(adapter_value)
        ).expanduser()

        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (
                PROJECT_ROOT / checkpoint_dir
            )
    else:
        checkpoint_dir = (
            output_root / f"epoch_{epoch}"
        )

    checkpoint_dir = checkpoint_dir.resolve()

    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"最佳 Checkpoint 目录不存在：{checkpoint_dir}"
        )

    return epoch, checkpoint_dir


def create_logger() -> logging.Logger:
    """创建控制台 Logger。"""

    logger = logging.getLogger(
        "training_manifest_evaluation"
    )
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )
    logger.addHandler(handler)

    return logger


def build_loader_arguments(
    checkpoint_config: Any,
    checkpoint_dir: Path,
    device: torch.device,
    attn_implementation: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """根据当前 Loader 的真实签名构造参数。

    这样兼容项目中 Loader 参数名的小幅差异，
    但不会绕过 Loader 自身的路径和 Token 校验。
    """

    signature = inspect.signature(
        load_qwen_ab_backend
    )
    arguments: dict[str, Any] = {}

    value_by_name = {
        "checkpoint": checkpoint_config,
        "checkpoint_config": checkpoint_config,
        "config": checkpoint_config,
        "reward_checkpoint": checkpoint_config,
        "checkpoint_dir": checkpoint_dir,
        "device": device,
        "device_str": str(device),
        "attn_implementation": (
            attn_implementation
        ),
        "logger": logger,
    }

    for name, parameter in (
        signature.parameters.items()
    ):
        if name in value_by_name:
            arguments[name] = value_by_name[name]
            continue

        if parameter.default is not inspect.Parameter.empty:
            continue

        raise RuntimeError(
            "无法自动构造 load_qwen_ab_backend "
            f"的必需参数：{name}；"
            f"当前签名={signature}"
        )

    return arguments


def extract_backend(
    loader_result: Any,
) -> Any:
    """从 Loader 返回值中提取 Backend。"""

    if hasattr(
        loader_result,
        "forward_ab_logits",
    ):
        return loader_result

    if isinstance(loader_result, tuple):
        for item in loader_result:
            if hasattr(
                item,
                "forward_ab_logits",
            ):
                return item

    raise TypeError(
        "load_qwen_ab_backend 未返回可识别的 Backend"
    )


def load_frozen_backend(
    checkpoint_config: Any,
    checkpoint_dir: Path,
    device: torch.device,
    attn_implementation: str,
    logger: logging.Logger,
) -> Any:
    """加载冻结的推理 Backend。"""

    loader_arguments = build_loader_arguments(
        checkpoint_config=checkpoint_config,
        checkpoint_dir=checkpoint_dir,
        device=device,
        attn_implementation=(
            attn_implementation
        ),
        logger=logger,
    )

    loader_result = load_qwen_ab_backend(
        **loader_arguments
    )
    backend = extract_backend(loader_result)

    model = getattr(backend, "model", None)

    if model is not None:
        model.eval()

        trainable_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        if trainable_count != 0:
            raise RuntimeError(
                "推理模型仍有可训练参数："
                f"{trainable_count}"
            )

    return backend


def choose_output_directory(
    requested_output_dir: Path | None,
    output_root: Path,
    epoch: int,
) -> Path:
    """确定评估输出目录。"""

    if requested_output_dir is not None:
        return resolve_project_path(
            requested_output_dir
        )

    return (
        output_root
        / f"train_evaluation_epoch_{epoch}"
    )


def print_metric(
    metrics: dict[str, Any],
    key: str,
    display_name: str,
) -> None:
    """打印存在的单项指标。"""

    value = metrics.get(key)

    if value is None:
        return

    if isinstance(value, float):
        print(f"{display_name}: {value:.6f}")
    else:
        print(f"{display_name}: {value}")


def print_summary(
    epoch: int,
    checkpoint_dir: Path,
    manifest_path: Path,
    metrics: dict[str, Any],
) -> None:
    """打印训练集评估摘要。"""

    print()
    print("========== TRAINING SET EVALUATION ==========")
    print(f"epoch: {epoch}")
    print(f"checkpoint: {checkpoint_dir}")
    print(f"manifest: {manifest_path}")

    print_metric(
        metrics,
        "num_samples",
        "samples",
    )
    print_metric(
        metrics,
        "roc_auc",
        "roc_auc",
    )
    print_metric(
        metrics,
        "accuracy",
        "accuracy",
    )
    print_metric(
        metrics,
        "balanced_accuracy",
        "balanced_accuracy",
    )
    print_metric(
        metrics,
        "pr_auc_positive",
        "pr_auc_positive",
    )
    print_metric(
        metrics,
        "pr_auc_negative",
        "pr_auc_negative",
    )
    print_metric(
        metrics,
        "brier",
        "brier",
    )
    print_metric(
        metrics,
        "ece",
        "ece",
    )
    print_metric(
        metrics,
        "mean_reward_margin",
        "mean_reward_margin",
    )
    print_metric(
        metrics,
        "mean_margin_pos_minus_neg",
        "mean_margin_pos_minus_neg",
    )


def main() -> None:
    """运行完整训练集评估。"""

    args = parse_args()
    logger = create_logger()

    output_root = resolve_project_path(
        args.output_root
    )
    manifest_path = resolve_project_path(
        args.manifest
    )

    epoch, checkpoint_dir = (
        read_best_checkpoint(output_root)
    )

    output_dir = choose_output_directory(
        requested_output_dir=args.output_dir,
        output_root=output_root,
        epoch=epoch,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 CUDA，但当前环境不可用"
            )

    device = torch.device(args.device)

    logger.info(
        "最佳 Epoch：%d",
        epoch,
    )
    logger.info(
        "Checkpoint：%s",
        checkpoint_dir,
    )
    logger.info(
        "训练清单：%s",
        manifest_path,
    )

    samples = read_preference_manifest(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        validate_image_paths=(
            not args.skip_image_path_check
        ),
    )

    logger.info(
        "样本数量：%d",
        len(samples),
    )
    logger.info(
        "标签分布：%s",
        count_labels(samples),
    )

    checkpoint_config = (
        load_reward_checkpoint_config(
            checkpoint_dir
        )
    )

    backend = load_frozen_backend(
        checkpoint_config=checkpoint_config,
        checkpoint_dir=checkpoint_dir,
        device=device,
        attn_implementation=(
            args.attn_implementation
        ),
        logger=logger,
    )

    metrics, predictions = evaluate_samples(
        backend=backend,
        samples=samples,
        batch_size=args.batch_size,
        negative_weight=float(
            checkpoint_config.negative_weight
        ),
        threshold=args.threshold,
        ece_bins=args.ece_bins,
    )

    metrics["evaluated_split"] = "train"
    metrics["epoch"] = epoch
    metrics["checkpoint_dir"] = str(
        checkpoint_dir
    )
    metrics["manifest_path"] = str(
        manifest_path
    )

    metrics_path = (
        output_dir / "train_metrics.json"
    )
    predictions_path = (
        output_dir / "train_predictions.jsonl"
    )

    write_json_atomic(
        metrics,
        metrics_path,
    )
    write_jsonl_atomic(
        predictions,
        predictions_path,
    )

    print_summary(
        epoch=epoch,
        checkpoint_dir=checkpoint_dir,
        manifest_path=manifest_path,
        metrics=metrics,
    )

    print()
    print(f"metrics: {metrics_path}")
    print(f"predictions: {predictions_path}")


if __name__ == "__main__":
    main()
