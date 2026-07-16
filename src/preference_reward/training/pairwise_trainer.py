#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Qwen3-VL Pairwise Reward Model 训练器。"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel

from preference_reward.common.config import (
    resolve_project_path,
)
from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)
from preference_reward.data.pairwise_manifest import (
    PairwiseSample,
    read_pairwise_manifest,
)
from preference_reward.models.qwen_ab_reward import (
    QwenABRewardBackend,
)
from preference_reward.models.qwen_pairwise_reward import (
    SCORE_TYPE_AB_LOGIT,
    SCORE_TYPE_SCALAR_HEAD,
    SUPPORTED_SCORE_TYPES,
    ScalarRewardHead,
    bradley_terry_loss,
    get_model_hidden_size,
    score_pair_batch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batched(
    values: Sequence[PairwiseSample],
    batch_size: int,
) -> Iterable[Sequence[PairwiseSample]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    for start in range(
        0,
        len(values),
        batch_size,
    ):
        yield values[start:start + batch_size]


def serializable_paths(
    paths: dict[str, Path],
) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in paths.items()
    }


def _set_learning_rate_scale(
    optimizer: torch.optim.Optimizer,
    base_learning_rates: list[float],
    scale: float,
) -> None:
    for group, base_lr in zip(
        optimizer.param_groups,
        base_learning_rates,
        strict=True,
    ):
        group["lr"] = float(base_lr) * scale


def _current_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> list[float]:
    return [
        float(group["lr"])
        for group in optimizer.param_groups
    ]


def _build_optimizer(
    backend: QwenABRewardBackend,
    reward_head: ScalarRewardHead | None,
    training_config: dict[str, Any],
) -> tuple[
    torch.optim.Optimizer,
    list[torch.nn.Parameter],
    list[float],
]:
    lora_parameters = [
        parameter
        for parameter in backend.model.parameters()
        if parameter.requires_grad
    ]

    if not lora_parameters:
        raise RuntimeError(
            "模型没有可训练 LoRA 参数"
        )

    model_lr = float(
        training_config["learning_rate"]
    )

    parameter_groups: list[dict[str, Any]] = [
        {
            "params": lora_parameters,
            "lr": model_lr,
            "name": "lora",
        }
    ]
    trainable_parameters = list(
        lora_parameters
    )
    base_learning_rates = [model_lr]

    if reward_head is not None:
        head_parameters = [
            parameter
            for parameter in reward_head.parameters()
            if parameter.requires_grad
        ]

        if not head_parameters:
            raise RuntimeError(
                "Reward Head 没有可训练参数"
            )

        head_lr = float(
            training_config.get(
                "reward_head_learning_rate",
                model_lr,
            )
        )

        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": head_lr,
                "name": "reward_head",
            }
        )
        trainable_parameters.extend(
            head_parameters
        )
        base_learning_rates.append(head_lr)

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(
            training_config["weight_decay"]
        ),
        betas=(
            float(training_config["adam_beta1"]),
            float(training_config["adam_beta2"]),
        ),
        eps=float(
            training_config["adam_epsilon"]
        ),
    )

    return (
        optimizer,
        trainable_parameters,
        base_learning_rates,
    )


def evaluate_pairwise_model(
    backend: QwenABRewardBackend,
    reward_head: ScalarRewardHead | None,
    score_type: str,
    samples: Sequence[PairwiseSample],
    batch_size: int,
    temperature: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """在严格 Pairwise 验证集上计算指标。"""

    backend.model.eval()

    if reward_head is not None:
        reward_head.eval()

    rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for pair_batch in batched(
            samples,
            batch_size,
        ):
            scores = score_pair_batch(
                backend=backend,
                pairs=pair_batch,
                score_type=score_type,
                reward_head=reward_head,
            )

            positive_scores = (
                scores.positive_scores
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            negative_scores = (
                scores.negative_scores
                .detach()
                .float()
                .cpu()
                .tolist()
            )

            for pair, positive, negative in zip(
                pair_batch,
                positive_scores,
                negative_scores,
                strict=True,
            ):
                margin = float(
                    positive - negative
                )
                nll = float(
                    F.softplus(
                        torch.tensor(
                            -margin / temperature,
                            dtype=torch.float64,
                        )
                    ).item()
                )

                rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "reference_image_path": str(
                            pair.reference_image
                        ),
                        "positive_image_path": str(
                            pair.positive_image
                        ),
                        "negative_image_path": str(
                            pair.negative_image
                        ),
                        "positive_reward_score": (
                            float(positive)
                        ),
                        "negative_reward_score": (
                            float(negative)
                        ),
                        "pairwise_margin": margin,
                        "pairwise_nll": nll,
                        "is_correct": margin > 0.0,
                        "is_tie": margin == 0.0,
                        "metadata": dict(pair.metadata),
                    }
                )

    margins = [
        float(row["pairwise_margin"])
        for row in rows
    ]
    nll_values = [
        float(row["pairwise_nll"])
        for row in rows
    ]
    correct_count = sum(
        bool(row["is_correct"])
        for row in rows
    )
    tie_count = sum(
        bool(row["is_tie"])
        for row in rows
    )

    metrics = {
        "pair_count": len(rows),
        "correct_count": int(correct_count),
        "incorrect_count": int(
            len(rows) - correct_count - tie_count
        ),
        "tie_count": int(tie_count),
        "pairwise_accuracy": float(
            correct_count / len(rows)
        ),
        "pairwise_nll": float(
            np.mean(nll_values)
        ),
        "mean_margin": float(
            np.mean(margins)
        ),
        "median_margin": float(
            median(margins)
        ),
        "minimum_margin": float(
            np.min(margins)
        ),
        "maximum_margin": float(
            np.max(margins)
        ),
    }

    return metrics, rows


def _checkpoint_is_better(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
) -> bool:
    """先比较 Accuracy，相同时比较 NLL。"""

    if best is None:
        return True

    candidate_accuracy = float(
        candidate["pairwise_accuracy"]
    )
    best_accuracy = float(
        best["pairwise_accuracy"]
    )

    if candidate_accuracy > best_accuracy:
        return True

    if candidate_accuracy < best_accuracy:
        return False

    return (
        float(candidate["pairwise_nll"])
        < float(best["pairwise_nll"])
    )


def _save_checkpoint(
    *,
    epoch: int,
    epoch_dir: Path,
    backend: QwenABRewardBackend,
    reward_head: ScalarRewardHead | None,
    score_type: str,
    resolved_paths: dict[str, Path],
    prompt_config: dict[str, Any],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    scalar_head_config: dict[str, Any],
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> None:
    epoch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backend.model.save_pretrained(
        str(epoch_dir)
    )

    head_path: Path | None = None

    if reward_head is not None:
        head_path = (
            epoch_dir / "reward_head.pt"
        )
        torch.save(
            reward_head.state_dict(),
            head_path,
        )

    checkpoint_config = {
        "checkpoint_format": (
            "qwen_pairwise_reward_v1"
        ),
        "epoch": int(epoch),
        "score_type": score_type,
        "base_model_path": str(
            resolved_paths["model_path"]
        ),
        "processor_path": str(
            resolved_paths["processor_dir"]
        ),
        "adapter_path": str(epoch_dir),
        "reward_head_path": (
            str(head_path)
            if head_path is not None
            else None
        ),
        "max_pixels": int(
            model_config["max_pixels"]
        ),
        "token_ids": {
            "A": int(backend.token_a),
            "B": int(backend.token_b),
        },
        "prompt": prompt_config,
        "scalar_head": scalar_head_config,
        "training": {
            "loss": "bradley_terry",
            "temperature": float(
                training_config["temperature"]
            ),
        },
        "metrics": metrics,
    }

    write_json_atomic(
        checkpoint_config,
        epoch_dir / "checkpoint_config.json",
    )
    write_json_atomic(
        metrics,
        epoch_dir / "epoch_metrics.json",
    )
    write_jsonl_atomic(
        predictions,
        epoch_dir / "val_pairwise_predictions.jsonl",
    )

    readme = (
        "# Pairwise Reward Checkpoint\n\n"
        f"- epoch: {epoch}\n"
        f"- score_type: {score_type}\n"
        f"- pairwise_accuracy: "
        f"{metrics['pairwise_accuracy']:.6f}\n"
        f"- pairwise_nll: "
        f"{metrics['pairwise_nll']:.6f}\n"
        f"- mean_margin: "
        f"{metrics['mean_margin']:.6f}\n"
    )
    (
        epoch_dir / "README.md"
    ).write_text(
        readme,
        encoding="utf-8",
    )


def run_pairwise_training(
    config: dict[str, Any],
    project_root: Path,
    logger: Any,
    log_path: Path,
    validate_image_paths: bool = True,
    limit_train: int = 0,
    limit_val: int = 0,
    debug_first_pair: bool = False,
) -> None:
    experiment_config = config["experiment"]
    paths_config = config["paths"]
    model_config = config["model"]
    prompt_config = config["prompt"]
    lora_config = config["lora"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]

    score_type = str(
        model_config["score_type"]
    )

    if score_type not in SUPPORTED_SCORE_TYPES:
        raise ValueError(
            f"不支持 score_type={score_type!r}"
        )

    seed = int(experiment_config["seed"])
    set_seed(seed)

    resolved_paths = {
        "model_path": resolve_project_path(
            project_root,
            paths_config["model_path"],
        ),
        "train_manifest": resolve_project_path(
            project_root,
            paths_config["train_manifest"],
        ),
        "val_manifest": resolve_project_path(
            project_root,
            paths_config["val_manifest"],
        ),
        "output_dir": resolve_project_path(
            project_root,
            paths_config["output_dir"],
        ),
    }
    resolved_paths["processor_dir"] = (
        resolved_paths["output_dir"]
        / "processor"
    )

    output_dir = resolved_paths["output_dir"]
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_samples = read_pairwise_manifest(
        manifest_path=resolved_paths[
            "train_manifest"
        ],
        project_root=project_root,
        validate_image_paths=(
            validate_image_paths
        ),
    )
    val_samples = read_pairwise_manifest(
        manifest_path=resolved_paths[
            "val_manifest"
        ],
        project_root=project_root,
        validate_image_paths=(
            validate_image_paths
        ),
    )

    if limit_train > 0:
        train_samples = train_samples[
            :limit_train
        ]

    if limit_val > 0:
        val_samples = val_samples[:limit_val]

    train_ids = {
        pair.pair_id
        for pair in train_samples
    }
    val_ids = {
        pair.pair_id
        for pair in val_samples
    }

    overlap = train_ids & val_ids

    if overlap:
        raise RuntimeError(
            "Train/Validation Pair ID 重叠："
            f"{sorted(overlap)}"
        )

    pair_batch_size = int(
        training_config["pair_batch_size"]
    )
    eval_pair_batch_size = int(
        evaluation_config[
            "pair_batch_size"
        ]
    )
    gradient_accumulation_steps = int(
        training_config[
            "gradient_accumulation_steps"
        ]
    )
    maximum_epochs = int(
        training_config["epochs"]
    )
    temperature = float(
        training_config["temperature"]
    )
    warmup_steps = int(
        training_config["warmup_steps"]
    )

    if pair_batch_size <= 0:
        raise ValueError(
            "pair_batch_size 必须大于 0"
        )

    if eval_pair_batch_size <= 0:
        raise ValueError(
            "evaluation.pair_batch_size 必须大于 0"
        )

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps 必须大于 0"
        )

    if maximum_epochs <= 0:
        raise ValueError(
            "epochs 必须大于 0"
        )

    if temperature <= 0:
        raise ValueError(
            "temperature 必须大于 0"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    backend = (
        QwenABRewardBackend.create_for_training(
            model_path=resolved_paths[
                "model_path"
            ],
            device=device,
            system_prompt=str(
                prompt_config["system"]
            ),
            user_prompt=str(
                prompt_config["user"]
            ),
            use_room_type=False,
            room_type_prefix="房型",
            max_pixels=int(
                model_config["max_pixels"]
            ),
            attn_implementation=str(
                model_config.get(
                    "attn_implementation",
                    "",
                )
            ),
            gradient_checkpointing=bool(
                model_config[
                    "gradient_checkpointing"
                ]
            ),
            lora_r=int(lora_config["r"]),
            lora_alpha=int(
                lora_config["alpha"]
            ),
            lora_dropout=float(
                lora_config["dropout"]
            ),
            target_modules=[
                str(item)
                for item in lora_config[
                    "target_modules"
                ]
            ],
            logger=logger,
        )
    )

    backend.save_processor(
        resolved_paths["processor_dir"]
    )

    scalar_head_config = dict(
        model_config.get(
            "scalar_head",
            {},
        )
    )
    reward_head: ScalarRewardHead | None = None

    if score_type == SCORE_TYPE_SCALAR_HEAD:
        hidden_size = get_model_hidden_size(
            backend.model
        )
        intermediate_size = int(
            scalar_head_config.get(
                "intermediate_size",
                1024,
            )
        )

        scalar_head_config = {
            "hidden_size": hidden_size,
            "intermediate_size": (
                intermediate_size
            ),
            "activation": "silu",
        }

        reward_head = ScalarRewardHead(
            hidden_size=hidden_size,
            intermediate_size=(
                intermediate_size
            ),
        ).to(device=device, dtype=torch.float32)

        logger.info(
            "Scalar Reward Head："
            "hidden=%d intermediate=%d",
            hidden_size,
            intermediate_size,
        )

    (
        optimizer,
        trainable_parameters,
        base_learning_rates,
    ) = _build_optimizer(
        backend=backend,
        reward_head=reward_head,
        training_config=training_config,
    )

    batches_per_epoch = math.ceil(
        len(train_samples) / pair_batch_size
    )
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch
        / gradient_accumulation_steps
    )
    maximum_optimizer_steps = (
        optimizer_steps_per_epoch
        * maximum_epochs
    )

    logger.info(
        "score_type=%s train_pairs=%d "
        "val_pairs=%d",
        score_type,
        len(train_samples),
        len(val_samples),
    )
    logger.info(
        "pair_batch=%d grad_accum=%d "
        "effective_pair_batch=%d",
        pair_batch_size,
        gradient_accumulation_steps,
        pair_batch_size
        * gradient_accumulation_steps,
    )
    logger.info(
        "optimizer_steps/epoch=%d "
        "maximum_optimizer_steps=%d "
        "warmup_steps=%d",
        optimizer_steps_per_epoch,
        maximum_optimizer_steps,
        warmup_steps,
    )

    if warmup_steps > 0:
        _set_learning_rate_scale(
            optimizer,
            base_learning_rates,
            1.0 / warmup_steps,
        )

    minimum_model_lr = float(
        training_config[
            "minimum_learning_rate"
        ]
    )
    minimum_head_lr = float(
        training_config.get(
            "minimum_reward_head_learning_rate",
            minimum_model_lr,
        )
    )
    minimum_lrs = [minimum_model_lr]

    if reward_head is not None:
        minimum_lrs.append(
            minimum_head_lr
        )

    plateau_config = training_config[
        "lr_scheduler"
    ]

    plateau_scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(
                plateau_config["factor"]
            ),
            patience=int(
                plateau_config["patience"]
            ),
            threshold=float(
                plateau_config["threshold"]
            ),
            min_lr=minimum_lrs,
        )
    )

    early_config = training_config[
        "early_stopping"
    ]
    early_enabled = bool(
        early_config["enabled"]
    )
    early_minimum_epochs = int(
        early_config["minimum_epochs"]
    )
    early_patience = int(
        early_config["patience"]
    )
    early_min_delta = float(
        early_config["min_delta"]
    )
    best_early_nll: float | None = None
    early_bad_epochs = 0

    run_config = {
        "experiment": experiment_config,
        "paths": serializable_paths(
            resolved_paths
        ),
        "model": {
            **model_config,
            "scalar_head": scalar_head_config,
        },
        "prompt": prompt_config,
        "lora": lora_config,
        "training": {
            **training_config,
            "effective_pair_batch_size": (
                pair_batch_size
                * gradient_accumulation_steps
            ),
            "optimizer_steps_per_epoch": (
                optimizer_steps_per_epoch
            ),
            "maximum_optimizer_steps": (
                maximum_optimizer_steps
            ),
        },
        "evaluation": evaluation_config,
        "dataset": {
            "train_pairs": len(train_samples),
            "val_pairs": len(val_samples),
        },
        "runtime": {
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "log_path": str(log_path),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    write_json_atomic(
        run_config,
        output_dir / "run_config.json",
    )

    history: dict[str, Any] = {
        "experiment": experiment_config[
            "name"
        ],
        "status": "running",
        "score_type": score_type,
        "last_completed_epoch": 0,
        "global_step": 0,
        "best_epoch": None,
        "best_metrics": None,
        "epochs": [],
    }
    history_path = (
        output_dir / "training_history.json"
    )
    write_json_atomic(
        history,
        history_path,
    )

    if debug_first_pair:
        backend.model.eval()
        if reward_head is not None:
            reward_head.eval()

        with torch.inference_mode():
            debug_scores = score_pair_batch(
                backend=backend,
                pairs=train_samples[:1],
                score_type=score_type,
                reward_head=reward_head,
            )

        logger.info(
            "DEBUG FIRST PAIR | pair_id=%s "
            "positive=%.6f negative=%.6f "
            "margin=%.6f",
            train_samples[0].pair_id,
            float(
                debug_scores
                .positive_scores[0]
                .item()
            ),
            float(
                debug_scores
                .negative_scores[0]
                .item()
            ),
            float(
                debug_scores.margins[0].item()
            ),
        )

    global_step = 0
    stopped_early = False
    best_metrics: dict[str, Any] | None = None

    try:
        for epoch in range(
            1,
            maximum_epochs + 1,
        ):
            epoch_started = time.perf_counter()
            epoch_pairs = list(train_samples)
            random.Random(
                seed + epoch
            ).shuffle(epoch_pairs)

            backend.model.train()
            if reward_head is not None:
                reward_head.train()

            optimizer.zero_grad(
                set_to_none=True
            )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            epoch_loss_sum = 0.0
            epoch_pair_count = 0
            epoch_correct_count = 0
            epoch_optimizer_steps = 0
            last_gradient_norm = 0.0

            for batch_index, pair_batch in enumerate(
                batched(
                    epoch_pairs,
                    pair_batch_size,
                ),
                start=1,
            ):
                group_start = (
                    (
                        batch_index - 1
                    )
                    // gradient_accumulation_steps
                ) * gradient_accumulation_steps + 1
                group_end = min(
                    group_start
                    + gradient_accumulation_steps
                    - 1,
                    batches_per_epoch,
                )
                group_size = (
                    group_end
                    - group_start
                    + 1
                )

                scores = score_pair_batch(
                    backend=backend,
                    pairs=pair_batch,
                    score_type=score_type,
                    reward_head=reward_head,
                )
                loss = bradley_terry_loss(
                    positive_scores=(
                        scores.positive_scores
                    ),
                    negative_scores=(
                        scores.negative_scores
                    ),
                    temperature=temperature,
                )

                (
                    loss / group_size
                ).backward()

                actual_pairs = len(pair_batch)
                epoch_loss_sum += (
                    float(loss.item())
                    * actual_pairs
                )
                epoch_pair_count += actual_pairs
                epoch_correct_count += int(
                    (
                        scores.margins.detach()
                        > 0
                    ).sum().item()
                )

                if batch_index == group_end:
                    gradient_norm = (
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters,
                            float(
                                training_config[
                                    "max_gradient_norm"
                                ]
                            ),
                        )
                    )
                    last_gradient_norm = float(
                        gradient_norm
                    )

                    optimizer.step()
                    optimizer.zero_grad(
                        set_to_none=True,
                    )
                    global_step += 1
                    epoch_optimizer_steps += 1

                    if (
                        warmup_steps > 0
                        and global_step
                        < warmup_steps
                    ):
                        _set_learning_rate_scale(
                            optimizer,
                            base_learning_rates,
                            (
                                global_step + 1
                            ) / warmup_steps,
                        )
                    elif (
                        warmup_steps > 0
                        and global_step
                        == warmup_steps
                    ):
                        _set_learning_rate_scale(
                            optimizer,
                            base_learning_rates,
                            1.0,
                        )

                    log_every = int(
                        training_config[
                            "log_every_optimizer_steps"
                        ]
                    )

                    if (
                        global_step == 1
                        or global_step
                        % log_every
                        == 0
                    ):
                        logger.info(
                            "epoch=%d step=%d "
                            "batch=%d/%d pairs=%d/%d "
                            "loss=%.6f avg_loss=%.6f "
                            "train_pair_acc=%.4f "
                            "grad_norm=%.6f lrs=%s",
                            epoch,
                            global_step,
                            batch_index,
                            batches_per_epoch,
                            epoch_pair_count,
                            len(epoch_pairs),
                            float(loss.item()),
                            (
                                epoch_loss_sum
                                / epoch_pair_count
                            ),
                            (
                                epoch_correct_count
                                / epoch_pair_count
                            ),
                            last_gradient_norm,
                            _current_learning_rates(
                                optimizer
                            ),
                        )

            val_metrics, val_predictions = (
                evaluate_pairwise_model(
                    backend=backend,
                    reward_head=reward_head,
                    score_type=score_type,
                    samples=val_samples,
                    batch_size=(
                        eval_pair_batch_size
                    ),
                    temperature=temperature,
                )
            )

            epoch_seconds = (
                time.perf_counter()
                - epoch_started
            )
            train_metrics = {
                "loss": float(
                    epoch_loss_sum
                    / epoch_pair_count
                ),
                "pairwise_accuracy": float(
                    epoch_correct_count
                    / epoch_pair_count
                ),
                "pair_count": int(
                    epoch_pair_count
                ),
                "optimizer_steps": int(
                    epoch_optimizer_steps
                ),
                "last_gradient_norm": float(
                    last_gradient_norm
                ),
            }
            epoch_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": val_metrics,
                "learning_rates": (
                    _current_learning_rates(
                        optimizer
                    )
                ),
                "epoch_seconds": float(
                    epoch_seconds
                ),
                "peak_cuda_memory_gb": (
                    float(
                        torch.cuda.max_memory_allocated()
                        / 1024**3
                    )
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }

            epoch_dir = (
                output_dir / f"epoch_{epoch}"
            )
            _save_checkpoint(
                epoch=epoch,
                epoch_dir=epoch_dir,
                backend=backend,
                reward_head=reward_head,
                score_type=score_type,
                resolved_paths=resolved_paths,
                prompt_config=prompt_config,
                model_config=model_config,
                training_config=training_config,
                scalar_head_config=(
                    scalar_head_config
                ),
                metrics=val_metrics,
                predictions=val_predictions,
            )

            if _checkpoint_is_better(
                val_metrics,
                best_metrics,
            ):
                best_metrics = dict(
                    val_metrics
                )
                history["best_epoch"] = epoch
                history["best_metrics"] = (
                    best_metrics
                )
                write_json_atomic(
                    {
                        "epoch": epoch,
                        "checkpoint_dir": str(
                            epoch_dir
                        ),
                        "selection_policy": (
                            "max pairwise_accuracy, "
                            "then min pairwise_nll"
                        ),
                        "metrics": best_metrics,
                    },
                    output_dir
                    / "best_checkpoint.json",
                )

            history["last_completed_epoch"] = (
                epoch
            )
            history["global_step"] = global_step
            history["epochs"].append(
                epoch_metrics
            )
            write_json_atomic(
                history,
                history_path,
            )

            logger.info(
                "EPOCH %d | train_loss=%.6f "
                "train_pair_acc=%.4f "
                "val_pair_acc=%.4f "
                "val_pair_nll=%.6f "
                "mean_margin=%.6f "
                "median_margin=%.6f "
                "time=%.1fs peak_cuda=%.2fGB",
                epoch,
                train_metrics["loss"],
                train_metrics[
                    "pairwise_accuracy"
                ],
                val_metrics[
                    "pairwise_accuracy"
                ],
                val_metrics[
                    "pairwise_nll"
                ],
                val_metrics["mean_margin"],
                val_metrics["median_margin"],
                epoch_seconds,
                epoch_metrics[
                    "peak_cuda_memory_gb"
                ],
            )

            # Warmup 完成后才允许 Plateau Scheduler 调整。
            if global_step >= warmup_steps:
                plateau_scheduler.step(
                    val_metrics[
                        "pairwise_nll"
                    ]
                )

            current_nll = float(
                val_metrics["pairwise_nll"]
            )

            if (
                best_early_nll is None
                or current_nll
                < best_early_nll
                - early_min_delta
            ):
                best_early_nll = current_nll
                early_bad_epochs = 0
            else:
                early_bad_epochs += 1

            if (
                early_enabled
                and epoch >= early_minimum_epochs
                and early_bad_epochs
                >= early_patience
            ):
                stopped_early = True
                logger.info(
                    "Early Stopping："
                    "val_pairwise_nll 连续 %d epoch "
                    "无有效改进",
                    early_bad_epochs,
                )
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        history["status"] = (
            "stopped_early"
            if stopped_early
            else "completed"
        )
        history["stopped_early"] = (
            stopped_early
        )
        write_json_atomic(
            history,
            history_path,
        )

    except Exception:
        history["status"] = "failed"
        history["last_error_at"] = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )
        write_json_atomic(
            history,
            history_path,
        )
        raise
