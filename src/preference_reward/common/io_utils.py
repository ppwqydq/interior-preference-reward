#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""项目通用文件写入工具。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def write_json_atomic(data: Any, output_path: Path) -> None:
    """原子写入 JSON，避免程序中断后留下不完整文件。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, output_path)

    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def write_jsonl_atomic(
    rows: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """原子写入 JSONL。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            for row in rows:
                file.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, output_path)

    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
