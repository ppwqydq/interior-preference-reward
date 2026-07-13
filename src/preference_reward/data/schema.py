#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""数据集构建过程中使用的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateSample:
    """从 CSV 中筛选出的候选训练样本。"""

    pair_id: str
    empty_room_url: str
    generated_furniture_url: str
    label: int


@dataclass(frozen=True)
class DownloadTask:
    """一张待下载图片。"""

    role: str
    url: str
    output_dir: Path


@dataclass(frozen=True)
class DownloadResult:
    """单张图片的下载结果。"""

    role: str
    url: str
    url_hash: str
    success: bool
    local_path: Path | None
    reused: bool
    error: str | None
