#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Qwen3-VL 双图偏好模型训练入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from preference_reward.common.config import (
    copy_config,
    load_yaml_config,
    resolve_project_path,
)
from preference_reward.common.logging_utils import (
    setup_logger,
)
from preference_reward.training.trainer import (
    run_training,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "训练 Qwen3-VL 空房间/家具布局偏好模型。"
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
        "--epochs",
        type=int,
        default=0,
        help="大于 0 时覆盖配置中的 epoch 数。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="覆盖配置中的输出目录。",
    )
    parser.add_argument(
        "--train_manifest",
        type=Path,
        default=None,
        help=(
            "覆盖配置中的训练清单。"
            "用于 OOF 时传入 inner_train.jsonl。"
        ),
    )
    parser.add_argument(
        "--val_manifest",
        type=Path,
        default=None,
        help=(
            "覆盖配置中的验证清单。"
            "用于 OOF 时传入 inner_val.jsonl。"
        ),
    )
    parser.add_argument(
        "--negative_weight",
        type=float,
        default=0.0,
        help=(
            "大于 0 时覆盖负样本类别权重。"
            "OOF 各折应使用同一个固定值。"
        ),
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help="覆盖实验名称。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="大于等于 0 时覆盖随机种子。",
    )
    parser.add_argument(
        "--limit_train",
        type=int,
        default=0,
        help="仅用于冒烟测试。",
    )
    parser.add_argument(
        "--limit_val",
        type=int,
        default=0,
        help="仅用于冒烟测试。",
    )
    parser.add_argument(
        "--debug_first_batch",
        action="store_true",
    )
    parser.add_argument(
        "--skip_image_path_check",
        action="store_true",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "只打印命令行覆盖后的配置，"
            "不加载模型也不启动训练。"
        ),
    )

    return parser.parse_args()


def apply_command_line_overrides(
    config: dict,
    args: argparse.Namespace,
) -> None:
    """把命令行参数覆盖到配置副本。"""

    if args.epochs > 0:
        config["training"]["epochs"] = (
            args.epochs
        )

    if args.output_dir:
        config["paths"]["output_dir"] = (
            args.output_dir
        )

    if args.train_manifest is not None:
        config["paths"]["train_manifest"] = str(
            args.train_manifest
        )

    if args.val_manifest is not None:
        config["paths"]["val_manifest"] = str(
            args.val_manifest
        )

    if args.negative_weight > 0:
        config["training"]["negative_weight"] = (
            args.negative_weight
        )

    if args.experiment_name:
        config["experiment"]["name"] = (
            args.experiment_name
        )

    if args.seed >= 0:
        config["experiment"]["seed"] = (
            args.seed
        )


def main() -> None:
    """加载配置并启动训练。"""

    args = parse_args()

    config = copy_config(
        load_yaml_config(
            args.config.resolve()
        )
    )

    apply_command_line_overrides(
        config=config,
        args=args,
    )

    if args.dry_run:
        print(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    log_dir = resolve_project_path(
        PROJECT_ROOT,
        config["paths"]["log_dir"],
    )

    logger, log_path = setup_logger(
        name="qwen_layout_training",
        log_dir=log_dir,
        prefix=str(
            config["experiment"]["name"]
        ),
    )

    logger.info(
        "项目根目录：%s",
        PROJECT_ROOT,
    )
    logger.info(
        "配置文件：%s",
        args.config.resolve(),
    )
    logger.info(
        "日志文件：%s",
        log_path,
    )
    logger.info(
        "训练清单：%s",
        config["paths"]["train_manifest"],
    )
    logger.info(
        "验证清单：%s",
        config["paths"]["val_manifest"],
    )
    logger.info(
        "输出目录：%s",
        config["paths"]["output_dir"],
    )

    run_training(
        config=config,
        project_root=PROJECT_ROOT,
        logger=logger,
        log_path=log_path,
        validate_image_paths=(
            not args.skip_image_path_check
        ),
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        debug_first_batch=(
            args.debug_first_batch
        ),
    )


if __name__ == "__main__":
    main()
