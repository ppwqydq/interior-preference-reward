#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""项目配置读取工具。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""

    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"配置文件根节点必须是对象：{path}")

    return config


def resolve_project_path(
    project_root: Path,
    value: str | Path,
) -> Path:
    """把配置中的相对路径转换为项目绝对路径。"""

    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (project_root / path).resolve()


def copy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """复制配置，避免命令行覆盖修改原始对象。"""

    return deepcopy(config)
