#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""项目日志工具。"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple


def setup_logger(
    name: str,
    log_dir: Path,
    prefix: str,
) -> Tuple[logging.Logger, Path]:
    """创建同时输出到终端和文件的日志器。"""

    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / (
        f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger, log_path
