#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Qwen3-VL 双图偏好模型训练入口。"""

from __future__ import annotations

import argparse
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

    return parser.parse_args()


def main() -> None:
    """加载配置并启动训练。"""

    args = parse_args()

    config = copy_config(
        load_yaml_config(
            args.config.resolve()
        )
    )

    if args.epochs > 0:
        config["training"]["epochs"] = (
            args.epochs
        )

    if args.output_dir:
        config["paths"]["output_dir"] = (
            args.output_dir
        )

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
