#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""评估 P1/P2 Pairwise Reward Checkpoint。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from preference_reward.common.config import (
    resolve_project_path,
)
from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)
from preference_reward.common.logging_utils import (
    setup_logger,
)
from preference_reward.data.pairwise_manifest import (
    read_pairwise_manifest,
)
from preference_reward.inference.pairwise_checkpoint import (
    load_pairwise_backend,
    load_pairwise_checkpoint_config,
)
from preference_reward.training.pairwise_trainer import (
    evaluate_pairwise_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pair_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="",
    )
    parser.add_argument(
        "--skip_image_path_check",
        action="store_true",
    )
    return parser.parse_args()


def bootstrap_accuracy_ci(
    rows: list[dict],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if iterations <= 0:
        raise ValueError(
            "bootstrap_iterations 必须大于 0"
        )

    correct = np.asarray(
        [
            1.0 if row["is_correct"] else 0.0
            for row in rows
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    values = np.empty(
        iterations,
        dtype=np.float64,
    )

    for index in range(iterations):
        sample_indices = rng.integers(
            0,
            len(correct),
            size=len(correct),
        )
        values[index] = float(
            correct[sample_indices].mean()
        )

    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def main() -> None:
    args = parse_args()

    checkpoint_dir = resolve_project_path(
        PROJECT_ROOT,
        args.checkpoint_dir,
    )
    manifest_path = resolve_project_path(
        PROJECT_ROOT,
        args.manifest,
    )
    output_dir = resolve_project_path(
        PROJECT_ROOT,
        args.output_dir,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger, _ = setup_logger(
        name="pairwise_checkpoint_evaluation",
        log_dir=output_dir,
        prefix="evaluate",
    )

    checkpoint = (
        load_pairwise_checkpoint_config(
            checkpoint_dir
        )
    )
    device = torch.device(args.device)
    backend, reward_head = (
        load_pairwise_backend(
            checkpoint=checkpoint,
            device=device,
            logger=logger,
            attn_implementation=(
                args.attn_implementation
            ),
        )
    )

    samples = read_pairwise_manifest(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        validate_image_paths=(
            not args.skip_image_path_check
        ),
    )
    temperature = float(
        checkpoint.raw
        .get("training", {})
        .get("temperature", 1.0)
    )

    metrics, rows = evaluate_pairwise_model(
        backend=backend,
        reward_head=reward_head,
        score_type=checkpoint.score_type,
        samples=samples,
        batch_size=args.pair_batch_size,
        temperature=temperature,
    )

    lower, upper = bootstrap_accuracy_ci(
        rows=rows,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    metrics[
        "bootstrap_accuracy_ci_95"
    ] = [lower, upper]
    metrics["checkpoint_dir"] = str(
        checkpoint_dir
    )
    metrics["checkpoint_epoch"] = (
        checkpoint.epoch
    )
    metrics["score_type"] = (
        checkpoint.score_type
    )
    metrics["manifest"] = str(
        manifest_path
    )

    write_json_atomic(
        metrics,
        output_dir / "metrics.json",
    )
    write_jsonl_atomic(
        rows,
        output_dir / "pairwise_predictions.jsonl",
    )

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
