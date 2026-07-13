#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""构建本地双图偏好训练数据集。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from preference_reward.common.logging_utils import setup_logger
from preference_reward.data.dataset_builder import build_dataset


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "筛选 thumbs up/dislike，下载空房间图和生成家具图，"
            "生成本地训练清单。"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "dataset.yaml",
        help="数据集配置文件。",
    )

    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置。"""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(
            f"配置文件根节点必须是对象：{path}"
        )

    return config


def resolve_path(value: str) -> Path:
    """把配置中的相对路径转换为项目绝对路径。"""

    path = Path(value).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main() -> None:
    """执行数据集构建。"""

    args = parse_args()
    config = load_config(args.config.resolve())

    paths = config["paths"]
    csv_config = config["csv"]
    columns = config["columns"]
    labels = config["labels"]
    download = config["download"]

    logger, log_path = setup_logger(
        name="build_dataset",
        log_dir=resolve_path(paths["log_dir"]),
        prefix="build_dataset",
    )

    # 让核心模块使用同一套日志处理器。
    import logging
    logging.getLogger(
        "preference_reward.data.dataset_builder"
    ).handlers = logger.handlers
    logging.getLogger(
        "preference_reward.data.dataset_builder"
    ).setLevel(logging.INFO)
    logging.getLogger(
        "preference_reward.data.dataset_builder"
    ).propagate = False

    logger.info("项目根目录：%s", PROJECT_ROOT)
    logger.info("配置文件：%s", args.config.resolve())
    logger.info("日志文件：%s", log_path)

    report = build_dataset(
        project_root=PROJECT_ROOT,
        raw_dir=resolve_path(paths["raw_dir"]),
        empty_room_dir=resolve_path(
            paths["empty_room_image_dir"]
        ),
        generated_furniture_dir=resolve_path(
            paths["generated_furniture_image_dir"]
        ),
        output_manifest=resolve_path(
            paths["output_manifest"]
        ),
        report_path=resolve_path(
            paths["build_report"]
        ),
        failed_downloads_path=resolve_path(
            paths["failed_downloads"]
        ),
        csv_pattern=str(
            csv_config["recursive_pattern"]
        ),
        csv_encoding=str(
            csv_config["encoding"]
        ),
        behavior_column=str(
            columns["behavior"]
        ),
        empty_room_column=str(
            columns["empty_room_url"]
        ),
        generated_furniture_column=str(
            columns["generated_furniture_url"]
        ),
        label_mapping=labels,
        workers=int(download["workers"]),
        connect_timeout=int(
            download["connect_timeout_seconds"]
        ),
        read_timeout=int(
            download["read_timeout_seconds"]
        ),
        retries=int(download["retries"]),
        max_image_bytes=int(
            download["max_image_size_mb"]
        ) * 1024 * 1024,
        user_agent=str(download["user_agent"]),
    )

    logger.info("========== 数据集构建完成 ==========")
    logger.info(
        "最终样本数：%d",
        report["final_samples"],
    )
    logger.info(
        "标签分布：%s",
        json.dumps(
            report["final_label_counts"],
            ensure_ascii=False,
        ),
    )
    logger.info(
        "新下载图片：%d",
        report["new_images_downloaded"],
    )
    logger.info(
        "复用已有图片：%d",
        report["existing_images_reused"],
    )
    logger.info(
        "下载失败图片：%d",
        report["failed_image_downloads"],
    )
    logger.info(
        "训练清单：%s",
        report["output_manifest"],
    )


if __name__ == "__main__":
    main()
