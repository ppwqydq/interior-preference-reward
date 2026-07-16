#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""反馈数据筛选、图片下载和训练清单生成。

处理规则：
1. 仅保留 thumbs up 和 dislike。
2. thumbs up 映射为 1，dislike 映射为 0。
3. 空房间图和生成家具图必须同时存在。
4. 同一图片对、同一标签只保留一次。
5. 同一图片对出现正负冲突时，整组排除。
6. 图片以 URL 的 SHA256 命名，后续运行自动复用。
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)
from preference_reward.data.schema import (
    CandidateSample,
    DownloadResult,
    DownloadTask,
)


LOGGER = logging.getLogger(__name__)

_THREAD_LOCAL = threading.local()

FORMAT_TO_EXTENSION = {
    "JPEG": ".jpg",
    "MPO": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}


def configure_csv_field_limit() -> None:
    """提高 CSV 单字段长度限制。

    原始 CSV 中包含较大的 Agent JSON 字段。即使最终不用这些字段，
    CSV 解析器仍然需要读取整行，因此需要提高字段长度上限。
    """

    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def sha256_text(text: str) -> str:
    """计算字符串的 SHA256。"""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def normalize_url(value: Any) -> str:
    """清理图片 URL 两侧空白。"""

    if value is None:
        return ""

    return str(value).strip()


def is_http_url(url: str) -> bool:
    """判断是否为 HTTP 或 HTTPS 链接。"""

    if not url:
        return False

    parsed = urlparse(url)

    return (
        parsed.scheme.lower() in {"http", "https"}
        and bool(parsed.netloc)
    )


def make_pair_id(
    empty_room_url: str,
    generated_furniture_url: str,
) -> str:
    """根据两张图片的 URL 生成稳定图片对标识。"""

    value = (
        empty_room_url
        + "\n"
        + generated_furniture_url
    )

    return sha256_text(value)


def find_csv_files(
    raw_dir: Path,
    pattern: str,
) -> List[Path]:
    """递归查找全部原始 CSV。"""

    files = sorted(
        path
        for path in raw_dir.rglob(pattern)
        if path.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"没有在目录中找到 CSV：{raw_dir}"
        )

    return files


def scan_csv_files(
    csv_files: Sequence[Path],
    encoding: str,
    behavior_column: str,
    empty_room_column: str,
    generated_furniture_column: str,
    label_mapping: Mapping[str, int],
) -> tuple[List[CandidateSample], Dict[str, Any]]:
    """扫描 CSV，并生成去重后的候选样本。

    此阶段只读取 URL，不下载图片。
    """

    configure_csv_field_limit()

    required_columns = {
        behavior_column,
        empty_room_column,
        generated_furniture_column,
    }

    behavior_counts: Counter[str] = Counter()
    label_counts_before_dedup: Counter[int] = Counter()

    total_rows = 0
    missing_image_rows = 0
    invalid_url_rows = 0
    duplicate_same_label_rows = 0

    # 同一个 pair_id 可能出现多个标签。
    labels_by_pair: Dict[str, set[int]] = defaultdict(set)

    # 同一 pair_id + label 只保存第一条代表记录。
    samples_by_pair_label: Dict[
        tuple[str, int],
        CandidateSample,
    ] = {}

    file_summaries: List[Dict[str, Any]] = []

    for csv_path in csv_files:
        file_rows = 0
        file_selected = 0

        with csv_path.open(
            "r",
            encoding=encoding,
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or [])

            missing_columns = required_columns - fieldnames

            if missing_columns:
                raise KeyError(
                    f"CSV 缺少必要字段：{csv_path}\n"
                    f"缺少：{sorted(missing_columns)}"
                )

            for row in reader:
                total_rows += 1
                file_rows += 1

                behavior = str(
                    row.get(behavior_column, "")
                ).strip().lower()

                behavior_counts[
                    behavior or "<EMPTY>"
                ] += 1

                if behavior not in label_mapping:
                    continue

                label = int(label_mapping[behavior])

                empty_room_url = normalize_url(
                    row.get(empty_room_column)
                )
                generated_furniture_url = normalize_url(
                    row.get(generated_furniture_column)
                )

                if (
                    not empty_room_url
                    or not generated_furniture_url
                ):
                    missing_image_rows += 1
                    continue

                if (
                    not is_http_url(empty_room_url)
                    or not is_http_url(
                        generated_furniture_url
                    )
                ):
                    invalid_url_rows += 1
                    continue

                pair_id = make_pair_id(
                    empty_room_url,
                    generated_furniture_url,
                )

                labels_by_pair[pair_id].add(label)

                key = (pair_id, label)

                if key in samples_by_pair_label:
                    duplicate_same_label_rows += 1
                    continue

                samples_by_pair_label[key] = CandidateSample(
                    pair_id=pair_id,
                    empty_room_url=empty_room_url,
                    generated_furniture_url=(
                        generated_furniture_url
                    ),
                    label=label,
                )

                label_counts_before_dedup[label] += 1
                file_selected += 1

        file_summaries.append(
            {
                "file": str(csv_path),
                "rows": file_rows,
                "new_unique_pair_label_records": file_selected,
            }
        )

    conflict_pair_ids = {
        pair_id
        for pair_id, labels in labels_by_pair.items()
        if len(labels) > 1
    }

    final_samples = sorted(
        (
            sample
            for sample in samples_by_pair_label.values()
            if sample.pair_id not in conflict_pair_ids
        ),
        key=lambda sample: sample.pair_id,
    )

    final_label_counts = Counter(
        sample.label
        for sample in final_samples
    )

    report = {
        "csv_files": file_summaries,
        "num_csv_files": len(csv_files),
        "total_csv_rows": total_rows,
        "behavior_counts": dict(
            sorted(behavior_counts.items())
        ),
        "missing_image_url_rows": missing_image_rows,
        "invalid_image_url_rows": invalid_url_rows,
        "duplicate_same_label_rows": (
            duplicate_same_label_rows
        ),
        "label_counts_before_conflict_removal": {
            str(key): int(value)
            for key, value in sorted(
                label_counts_before_dedup.items()
            )
        },
        "conflicting_image_pairs": len(
            conflict_pair_ids
        ),
        "candidate_samples_after_conflict_removal": len(
            final_samples
        ),
        "candidate_label_counts": {
            str(key): int(value)
            for key, value in sorted(
                final_label_counts.items()
            )
        },
        # 只保存少量冲突 ID，避免报告文件过大。
        "conflict_pair_id_examples": sorted(
            conflict_pair_ids
        )[:100],
    }

    return final_samples, report


def get_session(user_agent: str) -> requests.Session:
    """为每个下载线程创建独立的 HTTP Session。"""

    session = getattr(
        _THREAD_LOCAL,
        "session",
        None,
    )

    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": user_agent,
            }
        )
        _THREAD_LOCAL.session = session

    return session


def find_existing_image(
    output_dir: Path,
    url_hash: str,
) -> Path | None:
    """查找已经下载过的同一 URL 图片。"""

    candidates = sorted(
        path
        for path in output_dir.glob(
            f"{url_hash}.*"
        )
        if path.is_file()
        and not path.name.endswith(".part")
    )

    for path in candidates:
        try:
            with Image.open(path) as image:
                image.verify()
            return path
        except Exception:
            # 已存在但损坏的图片删除后重新下载。
            try:
                path.unlink()
            except OSError:
                pass

    return None


def validate_downloaded_image(
    path: Path,
) -> tuple[str, int, int]:
    """验证下载结果，并返回格式和尺寸。"""

    try:
        with Image.open(path) as image:
            image_format = str(
                image.format or ""
            ).upper()
            width, height = image.size
            image.verify()

    except UnidentifiedImageError as exc:
        raise ValueError(
            "响应内容不是可识别图片"
        ) from exc

    if not image_format:
        raise ValueError("无法识别图片格式")

    if width <= 0 or height <= 0:
        raise ValueError(
            f"图片尺寸无效：{width}x{height}"
        )

    return image_format, width, height


def download_image(
    task: DownloadTask,
    project_root: Path,
    connect_timeout: int,
    read_timeout: int,
    retries: int,
    max_image_bytes: int,
    user_agent: str,
) -> DownloadResult:
    """下载并验证一张图片。"""

    url_hash = sha256_text(task.url)
    task.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_path = find_existing_image(
        task.output_dir,
        url_hash,
    )

    if existing_path is not None:
        return DownloadResult(
            role=task.role,
            url=task.url,
            url_hash=url_hash,
            success=True,
            local_path=existing_path,
            reused=True,
            error=None,
        )

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        temp_path: Path | None = None

        try:
            session = get_session(user_agent)

            response = session.get(
                task.url,
                stream=True,
                timeout=(
                    connect_timeout,
                    read_timeout,
                ),
            )
            response.raise_for_status()

            content_length = response.headers.get(
                "Content-Length"
            )

            if (
                content_length
                and int(content_length) > max_image_bytes
            ):
                raise ValueError(
                    "图片超过大小限制："
                    f"{int(content_length)} bytes"
                )

            file_descriptor, temporary_name = (
                tempfile.mkstemp(
                    prefix=f".{url_hash}.",
                    suffix=".part",
                    dir=str(task.output_dir),
                )
            )
            temp_path = Path(temporary_name)

            downloaded_bytes = 0

            with os.fdopen(
                file_descriptor,
                "wb",
            ) as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    downloaded_bytes += len(chunk)

                    if downloaded_bytes > max_image_bytes:
                        raise ValueError(
                            "图片下载过程中超过大小限制："
                            f"{downloaded_bytes} bytes"
                        )

                    file.write(chunk)

                file.flush()
                os.fsync(file.fileno())

            if downloaded_bytes == 0:
                raise ValueError("下载内容为空")

            image_format, _, _ = (
                validate_downloaded_image(
                    temp_path
                )
            )

            extension = FORMAT_TO_EXTENSION.get(
                image_format,
                f".{image_format.lower()}",
            )

            final_path = (
                task.output_dir
                / f"{url_hash}{extension}"
            )

            os.replace(temp_path, final_path)
            temp_path = None

            return DownloadResult(
                role=task.role,
                url=task.url,
                url_hash=url_hash,
                success=True,
                local_path=final_path,
                reused=False,
                error=None,
            )

        except Exception as exc:
            last_error = exc

            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    pass

            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))

    return DownloadResult(
        role=task.role,
        url=task.url,
        url_hash=url_hash,
        success=False,
        local_path=None,
        reused=False,
        error=repr(last_error),
    )


def build_download_tasks(
    samples: Sequence[CandidateSample],
    empty_room_dir: Path,
    generated_furniture_dir: Path,
) -> List[DownloadTask]:
    """根据候选样本创建去重后的下载任务。"""

    tasks: Dict[
        tuple[str, str],
        DownloadTask,
    ] = {}

    for sample in samples:
        empty_key = (
            "empty_room",
            sample.empty_room_url,
        )
        generated_key = (
            "generated_furniture",
            sample.generated_furniture_url,
        )

        tasks.setdefault(
            empty_key,
            DownloadTask(
                role="empty_room",
                url=sample.empty_room_url,
                output_dir=empty_room_dir,
            ),
        )

        tasks.setdefault(
            generated_key,
            DownloadTask(
                role="generated_furniture",
                url=sample.generated_furniture_url,
                output_dir=generated_furniture_dir,
            ),
        )

    return sorted(
        tasks.values(),
        key=lambda task: (
            task.role,
            task.url,
        ),
    )


def download_all_images(
    tasks: Sequence[DownloadTask],
    project_root: Path,
    workers: int,
    connect_timeout: int,
    read_timeout: int,
    retries: int,
    max_image_bytes: int,
    user_agent: str,
) -> Dict[tuple[str, str], DownloadResult]:
    """并发下载所有唯一图片。"""

    results: Dict[
        tuple[str, str],
        DownloadResult,
    ] = {}

    completed = 0
    total = len(tasks)

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        future_to_task = {
            executor.submit(
                download_image,
                task,
                project_root,
                connect_timeout,
                read_timeout,
                retries,
                max_image_bytes,
                user_agent,
            ): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()

            results[
                (task.role, task.url)
            ] = result

            completed += 1

            if (
                completed == total
                or completed % 100 == 0
            ):
                LOGGER.info(
                    "图片处理进度：%d/%d",
                    completed,
                    total,
                )

    return results


def relative_project_path(
    path: Path,
    project_root: Path,
) -> str:
    """生成相对于项目根目录的 POSIX 路径。"""

    return path.resolve().relative_to(
        project_root.resolve()
    ).as_posix()


def build_dataset(
    project_root: Path,
    raw_dir: Path,
    empty_room_dir: Path,
    generated_furniture_dir: Path,
    output_manifest: Path,
    report_path: Path,
    failed_downloads_path: Path,
    csv_pattern: str,
    csv_encoding: str,
    behavior_column: str,
    empty_room_column: str,
    generated_furniture_column: str,
    label_mapping: Mapping[str, int],
    workers: int,
    connect_timeout: int,
    read_timeout: int,
    retries: int,
    max_image_bytes: int,
    user_agent: str,
) -> Dict[str, Any]:
    """执行完整的数据集构建流程。"""

    csv_files = find_csv_files(
        raw_dir,
        csv_pattern,
    )

    LOGGER.info(
        "发现 %d 个 CSV 文件",
        len(csv_files),
    )

    candidate_samples, scan_report = (
        scan_csv_files(
            csv_files=csv_files,
            encoding=csv_encoding,
            behavior_column=behavior_column,
            empty_room_column=empty_room_column,
            generated_furniture_column=(
                generated_furniture_column
            ),
            label_mapping={
                str(key).strip().lower(): int(value)
                for key, value in label_mapping.items()
            },
        )
    )

    LOGGER.info(
        "筛选、去重和冲突排除后候选样本：%d",
        len(candidate_samples),
    )

    tasks = build_download_tasks(
        samples=candidate_samples,
        empty_room_dir=empty_room_dir,
        generated_furniture_dir=(
            generated_furniture_dir
        ),
    )

    LOGGER.info(
        "需要检查或下载的唯一图片：%d",
        len(tasks),
    )

    download_results = download_all_images(
        tasks=tasks,
        project_root=project_root,
        workers=workers,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retries=retries,
        max_image_bytes=max_image_bytes,
        user_agent=user_agent,
    )

    manifest_rows: List[Dict[str, Any]] = []
    failed_samples = 0

    for sample in candidate_samples:
        empty_result = download_results[
            (
                "empty_room",
                sample.empty_room_url,
            )
        ]
        generated_result = download_results[
            (
                "generated_furniture",
                sample.generated_furniture_url,
            )
        ]

        if (
            not empty_result.success
            or not generated_result.success
            or empty_result.local_path is None
            or generated_result.local_path is None
        ):
            failed_samples += 1
            continue

        # 最终训练清单只保留模型需要的三个字段。
        manifest_rows.append(
            {
                "empty_room_image": (
                    relative_project_path(
                        empty_result.local_path,
                        project_root,
                    )
                ),
                "generated_furniture_image": (
                    relative_project_path(
                        generated_result.local_path,
                        project_root,
                    )
                ),
                "label": sample.label,
            }
        )

    manifest_rows.sort(
        key=lambda row: (
            row["empty_room_image"],
            row["generated_furniture_image"],
            row["label"],
        )
    )

    failed_download_rows = [
        {
            "role": result.role,
            "url": result.url,
            "url_hash": result.url_hash,
            "error": result.error,
        }
        for result in download_results.values()
        if not result.success
    ]

    write_jsonl_atomic(
        manifest_rows,
        output_manifest,
    )

    write_jsonl_atomic(
        sorted(
            failed_download_rows,
            key=lambda row: (
                row["role"],
                row["url_hash"],
            ),
        ),
        failed_downloads_path,
    )

    successful_downloads = sum(
        1
        for result in download_results.values()
        if result.success and not result.reused
    )
    reused_downloads = sum(
        1
        for result in download_results.values()
        if result.success and result.reused
    )

    final_label_counts = Counter(
        int(row["label"])
        for row in manifest_rows
    )

    final_report = {
        **scan_report,
        "unique_download_tasks": len(tasks),
        "new_images_downloaded": successful_downloads,
        "existing_images_reused": reused_downloads,
        "failed_image_downloads": len(
            failed_download_rows
        ),
        "samples_removed_due_to_download_failure": (
            failed_samples
        ),
        "final_samples": len(manifest_rows),
        "final_label_counts": {
            str(key): int(value)
            for key, value in sorted(
                final_label_counts.items()
            )
        },
        "output_manifest": str(
            output_manifest.resolve()
        ),
        "failed_downloads_file": str(
            failed_downloads_path.resolve()
        ),
    }

    write_json_atomic(
        final_report,
        report_path,
    )

    return final_report
