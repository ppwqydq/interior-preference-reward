#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""训练 Qwen3-VL Pairwise Reward Model。"""

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
from preference_reward.training.pairwise_trainer import (
    run_pairwise_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "训练 Qwen3-VL Pairwise Reward Model。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--train_manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--val_manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--limit_train",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--limit_val",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--debug_first_pair",
        action="store_true",
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


def apply_overrides(
    config: dict,
    args: argparse.Namespace,
) -> None:
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
    if args.experiment_name:
        config["experiment"]["name"] = (
            args.experiment_name
        )
    if args.seed >= 0:
        config["experiment"]["seed"] = (
            args.seed
        )


def main() -> None:
    args = parse_args()
    config = copy_config(
        load_yaml_config(
            args.config.resolve()
        )
    )
    apply_overrides(config, args)

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
        name="qwen_pairwise_training",
        log_dir=log_dir,
        prefix=str(
            config["experiment"]["name"]
        ),
    )

    logger.info(
        "配置文件：%s",
        args.config.resolve(),
    )
    logger.info(
        "训练 Pair：%s",
        config["paths"]["train_manifest"],
    )
    logger.info(
        "验证 Pair：%s",
        config["paths"]["val_manifest"],
    )

    run_pairwise_training(
        config=config,
        project_root=PROJECT_ROOT,
        logger=logger,
        log_path=log_path,
        validate_image_paths=(
            not args.skip_image_path_check
        ),
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        debug_first_pair=(
            args.debug_first_pair
        ),
    )


if __name__ == "__main__":
    main()
