#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""依次执行 OOF 各折训练。

本脚本只负责编排现有 ``scripts/train.py``：

1. Fold 1 成功结束后才启动 Fold 2；
2. 每折使用 inner_train.jsonl 和 inner_val.jsonl；
3. outer_holdout.jsonl 不参与训练和早停；
4. 已成功完成的 Fold 默认跳过；
5. 失败或不完整的输出目录默认拒绝覆盖，使用 ``--force`` 明确重跑。

本脚本不负责 Outer Holdout 推理和 OOF 预测合并。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
SUCCESS_STATUSES = {"completed", "early_stopped"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="按照 Fold 编号依次启动 OOF 训练。"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "qwen8b_layout_512.yaml"
        ),
        help="基础训练配置。",
    )
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "oof_4fold"
        ),
        help="OOF 划分根目录。",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "qwen3_vl_8b_layout_ab_512_oof"
        ),
        help="所有 Fold 的训练输出根目录。",
    )
    parser.add_argument(
        "--start_fold",
        type=int,
        default=1,
        help="起始 Fold，包含该 Fold。",
    )
    parser.add_argument(
        "--end_fold",
        type=int,
        default=4,
        help="结束 Fold，包含该 Fold。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="每折最大训练 Epoch。",
    )
    parser.add_argument(
        "--negative_weight",
        type=float,
        default=2.057142857142857,
        help="所有 Fold 共用的固定负样本权重。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="默认所有 Fold 使用同一个随机种子。",
    )
    parser.add_argument(
        "--vary_seed_by_fold",
        action="store_true",
        help="使用 seed + fold - 1 作为每折随机种子。",
    )
    parser.add_argument(
        "--experiment_prefix",
        type=str,
        default="qwen3_vl_8b_layout_ab_512_oof",
        help="每折实验名称前缀。",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=3.0,
        help="相邻 Fold 之间等待的秒数。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "删除所选 Fold 的已有输出并从头重跑。"
            "Trainer 当前不支持断点续训。"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只校验并打印各折训练命令。",
    )
    parser.add_argument(
        "--skip_image_path_check",
        action="store_true",
        help="传递给 train.py；正式训练通常不建议使用。",
    )
    parser.add_argument(
        "--debug_first_batch",
        action="store_true",
        help="传递给 train.py。",
    )

    return parser.parse_args()


def now_text() -> str:
    """返回秒级时间字符串。"""

    return datetime.now().isoformat(timespec="seconds")


def write_json_atomic(data: Any, path: Path) -> None:
    """原子写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as file:
        temp_path = Path(file.name)
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.flush()

    temp_path.replace(path)


def load_json(path: Path) -> dict[str, Any] | None:
    """读取 JSON；文件不存在时返回 None。"""

    if not path.is_file():
        return None

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None


def completed_fold_info(
    output_dir: Path,
) -> dict[str, Any] | None:
    """判断 Fold 是否已成功完成。"""

    history = load_json(
        output_dir / "training_history.json"
    )
    best = load_json(
        output_dir / "best_checkpoint.json"
    )

    if history is None or best is None:
        return None

    status = str(history.get("status", ""))

    if status not in SUCCESS_STATUSES:
        return None

    adapter_value = best.get("adapter_path")

    if not adapter_value:
        return None

    adapter_path = Path(str(adapter_value)).expanduser()

    if not adapter_path.is_absolute():
        adapter_path = (
            PROJECT_ROOT / adapter_path
        ).resolve()

    if not adapter_path.is_dir():
        return None

    if not (
        adapter_path / "adapter_config.json"
    ).is_file():
        return None

    if not (
        adapter_path / "adapter_model.safetensors"
    ).is_file():
        return None

    return {
        "status": status,
        "best_epoch": history.get("best_epoch"),
        "best_value": history.get("best_value"),
        "adapter_path": str(adapter_path),
    }


def validate_args(args: argparse.Namespace) -> None:
    """校验流水线参数和基础文件。"""

    if args.start_fold <= 0:
        raise ValueError("start_fold 必须大于 0")

    if args.end_fold < args.start_fold:
        raise ValueError(
            "end_fold 必须大于等于 start_fold"
        )

    if args.epochs <= 0:
        raise ValueError("epochs 必须大于 0")

    if args.negative_weight <= 0:
        raise ValueError(
            "negative_weight 必须大于 0"
        )

    if args.seed < 0:
        raise ValueError("seed 必须大于等于 0")

    if args.sleep_seconds < 0:
        raise ValueError(
            "sleep_seconds 必须大于等于 0"
        )

    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"训练入口不存在：{TRAIN_SCRIPT}"
        )

    config_path = args.config.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"训练配置不存在：{config_path}"
        )

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"OOF 划分目录不存在：{split_dir}"
        )

    for fold in range(
        args.start_fold,
        args.end_fold + 1,
    ):
        fold_dir = split_dir / f"fold_{fold}"

        for filename in (
            "inner_train.jsonl",
            "inner_val.jsonl",
            "outer_holdout.jsonl",
        ):
            path = fold_dir / filename

            if not path.is_file():
                raise FileNotFoundError(
                    f"Fold {fold} 缺少文件：{path}"
                )


def build_command(
    args: argparse.Namespace,
    fold: int,
) -> list[str]:
    """构造单个 Fold 的训练命令。"""

    split_dir = args.split_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    fold_split_dir = split_dir / f"fold_{fold}"
    fold_output_dir = output_root / f"fold_{fold}"

    fold_seed = (
        args.seed + fold - 1
        if args.vary_seed_by_fold
        else args.seed
    )

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(args.config.expanduser().resolve()),
        "--train_manifest",
        str(fold_split_dir / "inner_train.jsonl"),
        "--val_manifest",
        str(fold_split_dir / "inner_val.jsonl"),
        "--output_dir",
        str(fold_output_dir),
        "--experiment_name",
        f"{args.experiment_prefix}_fold_{fold}",
        "--negative_weight",
        str(args.negative_weight),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(fold_seed),
    ]

    if args.skip_image_path_check:
        command.append("--skip_image_path_check")

    if args.debug_first_batch:
        command.append("--debug_first_batch")

    return command


def command_text(command: list[str]) -> str:
    """生成便于复制查看的命令文本。"""

    return " ".join(
        subprocess.list2cmdline([part])
        for part in command
    )


def main() -> None:
    """依次执行选定的 Fold。"""

    args = parse_args()
    validate_args(args)

    output_root = args.output_root.expanduser().resolve()

    commands = {
        fold: build_command(args, fold)
        for fold in range(
            args.start_fold,
            args.end_fold + 1,
        )
    }

    print("========== OOF 顺序训练计划 ==========")
    print(
        f"Fold 范围：{args.start_fold}..{args.end_fold}"
    )
    print(f"每折最大 Epoch：{args.epochs}")
    print(
        "固定 negative_weight："
        f"{args.negative_weight:.15g}"
    )
    print(
        "随机种子策略："
        + (
            "按 Fold 递增"
            if args.vary_seed_by_fold
            else "所有 Fold 相同"
        )
    )
    print(f"输出根目录：{output_root}")

    for fold, command in commands.items():
        print()
        print(f"Fold {fold}:")
        print(command_text(command))

    if args.dry_run:
        print()
        print("dry_run 完成，未启动训练。")
        return

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_path = output_root / "pipeline_status.json"
    state: dict[str, Any] = {
        "status": "running",
        "started_at": now_text(),
        "finished_at": None,
        "config": str(
            args.config.expanduser().resolve()
        ),
        "split_dir": str(
            args.split_dir.expanduser().resolve()
        ),
        "output_root": str(output_root),
        "start_fold": args.start_fold,
        "end_fold": args.end_fold,
        "epochs": args.epochs,
        "negative_weight": args.negative_weight,
        "seed": args.seed,
        "vary_seed_by_fold": (
            args.vary_seed_by_fold
        ),
        "folds": {},
    }
    write_json_atomic(state, state_path)

    try:
        for fold in range(
            args.start_fold,
            args.end_fold + 1,
        ):
            fold_output_dir = (
                output_root / f"fold_{fold}"
            )

            completed_info = completed_fold_info(
                fold_output_dir
            )

            if completed_info is not None and not args.force:
                print()
                print(
                    f"Fold {fold} 已完成，跳过："
                    f"status={completed_info['status']}，"
                    f"best_epoch={completed_info['best_epoch']}，"
                    f"best_auc={completed_info['best_value']}"
                )
                state["folds"][str(fold)] = {
                    "status": "skipped_completed",
                    "checked_at": now_text(),
                    **completed_info,
                }
                write_json_atomic(
                    state,
                    state_path,
                )
                continue

            if fold_output_dir.exists():
                if args.force:
                    print()
                    print(
                        f"删除 Fold {fold} 旧输出："
                        f"{fold_output_dir}"
                    )
                    shutil.rmtree(fold_output_dir)
                elif any(fold_output_dir.iterdir()):
                    raise RuntimeError(
                        f"Fold {fold} 输出目录已存在但未确认完成："
                        f"{fold_output_dir}\n"
                        "Trainer 不支持断点续训。"
                        "确认需要从头重跑后使用 --force。"
                    )

            command = commands[fold]
            state["folds"][str(fold)] = {
                "status": "running",
                "started_at": now_text(),
                "command": command,
                "output_dir": str(fold_output_dir),
            }
            write_json_atomic(state, state_path)

            print()
            print(
                "========================================"
            )
            print(f"开始训练 Fold {fold}")
            print(
                "========================================"
            )

            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
            )

            if result.returncode != 0:
                state["status"] = "failed"
                state["finished_at"] = now_text()
                state["folds"][str(fold)].update(
                    {
                        "status": "failed",
                        "finished_at": now_text(),
                        "returncode": result.returncode,
                    }
                )
                write_json_atomic(
                    state,
                    state_path,
                )
                raise RuntimeError(
                    f"Fold {fold} 训练失败，"
                    f"returncode={result.returncode}。"
                    "流水线已停止，不会启动下一折。"
                )

            completed_info = completed_fold_info(
                fold_output_dir
            )

            if completed_info is None:
                state["status"] = "failed"
                state["finished_at"] = now_text()
                state["folds"][str(fold)].update(
                    {
                        "status": "verification_failed",
                        "finished_at": now_text(),
                        "returncode": result.returncode,
                    }
                )
                write_json_atomic(
                    state,
                    state_path,
                )
                raise RuntimeError(
                    f"Fold {fold} 进程返回成功，"
                    "但训练完成文件校验失败。"
                    "流水线已停止。"
                )

            state["folds"][str(fold)].update(
                {
                    "status": "succeeded",
                    "finished_at": now_text(),
                    "returncode": result.returncode,
                    **completed_info,
                }
            )
            write_json_atomic(state, state_path)

            print()
            print(
                f"Fold {fold} 完成："
                f"status={completed_info['status']}，"
                f"best_epoch={completed_info['best_epoch']}，"
                f"best_auc={completed_info['best_value']}"
            )

            if (
                fold < args.end_fold
                and args.sleep_seconds > 0
            ):
                print(
                    f"等待 {args.sleep_seconds:g} 秒后"
                    "启动下一折。"
                )
                time.sleep(args.sleep_seconds)

        state["status"] = "completed"
        state["finished_at"] = now_text()
        write_json_atomic(state, state_path)

        print()
        print("========== OOF 四折训练全部完成 ==========")
        print(f"流水线状态：{state_path}")

    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["finished_at"] = now_text()
        write_json_atomic(state, state_path)
        print("\n收到中断信号，流水线已停止。")
        raise
    except Exception:
        if state.get("status") == "running":
            state["status"] = "failed"
            state["finished_at"] = now_text()
            write_json_atomic(state, state_path)
        raise


if __name__ == "__main__":
    main()
