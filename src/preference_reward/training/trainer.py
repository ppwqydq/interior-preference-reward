#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Qwen3-VL 双图偏好训练器。

当前控制策略：

- 固定 optimizer step 数的线性 Warmup；
- 验证 ROC-AUC 停滞时降低学习率；
- 验证 ROC-AUC 长期停滞时提前停止；
- 每个 epoch 保存一份 LoRA adapter；
- best_checkpoint.json 仅记录最佳 epoch 的路径，不复制模型文件。
"""

from __future__ import annotations

import math
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from preference_reward.common.config import (
    resolve_project_path,
)
from preference_reward.common.io_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)
from preference_reward.data.manifest import (
    PreferenceSample,
    batched,
    count_labels,
    read_preference_manifest,
)
from preference_reward.evaluation.classification import (
    evaluate_samples,
)
from preference_reward.models.qwen_ab_reward import (
    QwenABRewardBackend,
)
from preference_reward.training.margin_regularization import (
    build_margin_regularized_loss,
)
from preference_reward.training.scheduler import (
    EarlyStoppingController,
    LinearWarmupController,
    PlateauLrController,
    get_optimizer_learning_rate,
)


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 Torch 随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_negative_weight(
    samples: List[PreferenceSample],
    configured_value: Any,
) -> float:
    """计算负样本类别权重。"""

    counts = Counter(
        sample.label
        for sample in samples
    )

    positive_count = int(counts.get(1, 0))
    negative_count = int(counts.get(0, 0))

    if positive_count == 0 or negative_count == 0:
        raise RuntimeError(
            "训练集必须同时包含正负样本"
        )

    if (
        isinstance(configured_value, str)
        and configured_value.strip().lower()
        == "auto"
    ):
        return (
            positive_count
            / negative_count
        )

    value = float(configured_value)

    if value <= 0:
        raise ValueError(
            "negative_weight 必须大于 0"
        )

    return value


def serializable_paths(
    paths: Dict[str, Path],
) -> Dict[str, str]:
    """将 Path 转换为可写入 JSON 的字符串。"""

    return {
        key: str(value)
        for key, value in paths.items()
    }


def is_valid_monitor_value(
    value: Any,
) -> bool:
    """判断验证指标是否可以参与调度和早停。"""

    if value is None:
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(numeric_value)


def run_training(
    config: Dict[str, Any],
    project_root: Path,
    logger: Any,
    log_path: Path,
    validate_image_paths: bool = True,
    limit_train: int = 0,
    limit_val: int = 0,
    debug_first_batch: bool = False,
) -> None:
    """执行完整训练流程。"""

    experiment_config = config["experiment"]
    paths_config = config["paths"]
    model_config = config["model"]
    prompt_config = config["prompt"]
    lora_config = config["lora"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]

    lr_scheduler_config = training_config[
        "lr_scheduler"
    ]
    early_stopping_config = training_config[
        "early_stopping"
    ]

    lr_monitor_name = str(
        lr_scheduler_config["monitor"]
    )
    early_monitor_name = str(
        early_stopping_config["monitor"]
    )

    if lr_monitor_name != early_monitor_name:
        raise ValueError(
            "当前实现要求学习率调度和早停使用同一个监控指标"
        )

    monitor_name = lr_monitor_name

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

    output_dir = resolved_paths["output_dir"]
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "读取训练集：%s",
        resolved_paths["train_manifest"],
    )
    train_samples = read_preference_manifest(
        manifest_path=resolved_paths[
            "train_manifest"
        ],
        project_root=project_root,
        validate_image_paths=validate_image_paths,
    )

    logger.info(
        "读取验证集：%s",
        resolved_paths["val_manifest"],
    )
    val_samples = read_preference_manifest(
        manifest_path=resolved_paths[
            "val_manifest"
        ],
        project_root=project_root,
        validate_image_paths=validate_image_paths,
    )

    if limit_train > 0:
        train_samples = train_samples[
            :limit_train
        ]

    if limit_val > 0:
        val_samples = val_samples[
            :limit_val
        ]

    batch_size = int(
        training_config["batch_size"]
    )
    eval_batch_size = int(
        training_config["eval_batch_size"]
    )
    gradient_accumulation_steps = int(
        training_config[
            "gradient_accumulation_steps"
        ]
    )
    maximum_epochs = int(
        training_config["epochs"]
    )

    if batch_size <= 0:
        raise ValueError(
            "batch_size 必须大于 0"
        )

    if eval_batch_size <= 0:
        raise ValueError(
            "eval_batch_size 必须大于 0"
        )

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps 必须大于 0"
        )

    if maximum_epochs <= 0:
        raise ValueError(
            "epochs 必须大于 0"
        )

    negative_weight = (
        calculate_negative_weight(
            train_samples,
            training_config[
                "negative_weight"
            ],
        )
    )

    margin_l2_weight = float(
        training_config.get(
            "margin_l2_weight",
            0.0,
        )
    )

    if margin_l2_weight < 0:
        raise ValueError(
            "margin_l2_weight 不能小于 0"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        logger.warning(
            "CUDA 不可用，8B 模型训练将非常慢"
        )

    batches_per_epoch = math.ceil(
        len(train_samples) / batch_size
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
        "训练样本：%d，标签分布：%s",
        len(train_samples),
        count_labels(train_samples),
    )
    logger.info(
        "验证样本：%d，标签分布：%s",
        len(val_samples),
        count_labels(val_samples),
    )
    logger.info(
        "negative_weight：%.10f",
        negative_weight,
    )
    logger.info(
        "margin_l2_weight：%.10f",
        margin_l2_weight,
    )
    logger.info(
        "batch=%d，grad_accum=%d，effective_batch=%d",
        batch_size,
        gradient_accumulation_steps,
        batch_size
        * gradient_accumulation_steps,
    )
    logger.info(
        "optimizer steps/epoch=%d，最多 optimizer steps=%d",
        optimizer_steps_per_epoch,
        maximum_optimizer_steps,
    )
    logger.info(
        "最多 epoch=%d，Warmup steps=%d",
        maximum_epochs,
        int(training_config["warmup_steps"]),
    )
    logger.info(
        "监控指标=%s，Plateau patience=%d，Early Stop patience=%d",
        monitor_name,
        int(lr_scheduler_config["patience"]),
        int(early_stopping_config["patience"]),
    )
    logger.info(
        "max_pixels=%d",
        int(model_config["max_pixels"]),
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
            use_room_type=bool(
                prompt_config.get(
                    "use_room_type",
                    False,
                )
            ),
            room_type_prefix=str(
                prompt_config.get(
                    "room_type_prefix",
                    "房型",
                )
            ),
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
                str(value)
                for value in lora_config[
                    "target_modules"
                ]
            ],
            logger=logger,
        )
    )

    processor_dir = (
        output_dir / "processor"
    )
    backend.save_processor(
        processor_dir
    )

    trainable_parameters = [
        parameter
        for parameter in backend.model.parameters()
        if parameter.requires_grad
    ]

    peak_learning_rate = float(
        training_config["learning_rate"]
    )
    minimum_learning_rate = float(
        training_config[
            "minimum_learning_rate"
        ]
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=peak_learning_rate,
        weight_decay=float(
            training_config[
                "weight_decay"
            ]
        ),
        betas=(
            float(
                training_config[
                    "adam_beta1"
                ]
            ),
            float(
                training_config[
                    "adam_beta2"
                ]
            ),
        ),
        eps=float(
            training_config[
                "adam_epsilon"
            ]
        ),
    )

    warmup_controller = (
        LinearWarmupController(
            base_learning_rate=(
                peak_learning_rate
            ),
            warmup_steps=int(
                training_config[
                    "warmup_steps"
                ]
            ),
        )
    )

    initial_learning_rate = (
        warmup_controller.initialize(
            optimizer
        )
    )

    plateau_controller = (
        PlateauLrController(
            mode=str(
                lr_scheduler_config["mode"]
            ),
            factor=float(
                lr_scheduler_config["factor"]
            ),
            patience=int(
                lr_scheduler_config[
                    "patience"
                ]
            ),
            threshold=float(
                lr_scheduler_config[
                    "threshold"
                ]
            ),
            minimum_learning_rate=(
                minimum_learning_rate
            ),
        )
    )

    early_stopping_controller = (
        EarlyStoppingController(
            enabled=bool(
                early_stopping_config[
                    "enabled"
                ]
            ),
            mode=str(
                early_stopping_config["mode"]
            ),
            minimum_epochs=int(
                early_stopping_config[
                    "minimum_epochs"
                ]
            ),
            patience=int(
                early_stopping_config[
                    "patience"
                ]
            ),
            min_delta=float(
                early_stopping_config[
                    "min_delta"
                ]
            ),
        )
    )

    logger.info(
        "第一个 optimizer step 学习率：%.10g",
        initial_learning_rate,
    )

    run_config = {
        "experiment": experiment_config,
        "paths": serializable_paths(
            resolved_paths
        ),
        "model": model_config,
        "prompt": prompt_config,
        "lora": lora_config,
        "training": {
            **training_config,
            "negative_weight_resolved": (
                negative_weight
            ),
            "effective_batch_size": (
                batch_size
                * gradient_accumulation_steps
            ),
            "batches_per_epoch": (
                batches_per_epoch
            ),
            "optimizer_steps_per_epoch": (
                optimizer_steps_per_epoch
            ),
            "maximum_optimizer_steps": (
                maximum_optimizer_steps
            ),
            "initial_warmup_learning_rate": (
                initial_learning_rate
            ),
        },
        "evaluation": evaluation_config,
        "dataset": {
            "train_samples": len(
                train_samples
            ),
            "train_label_counts": (
                count_labels(train_samples)
            ),
            "val_samples": len(
                val_samples
            ),
            "val_label_counts": (
                count_labels(val_samples)
            ),
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
        "checkpoint_policy": {
            "save_every_epoch": True,
            "best_checkpoint_pointer": True,
            "copy_best_adapter": False,
            "automatic_last_checkpoint": False,
            "save_optimizer_state": False,
        },
    }

    write_json_atomic(
        run_config,
        output_dir / "run_config.json",
    )

    history: Dict[str, Any] = {
        "experiment": experiment_config[
            "name"
        ],
        "status": "running",
        "monitor": monitor_name,
        "last_completed_epoch": 0,
        "global_step": 0,
        "best_epoch": None,
        "best_value": None,
        "epochs": [],
    }

    history_path = (
        output_dir
        / "training_history.json"
    )
    best_checkpoint_path = (
        output_dir
        / "best_checkpoint.json"
    )

    write_json_atomic(
        history,
        history_path,
    )

    if debug_first_batch:
        debug_samples = train_samples[
            :batch_size
        ]

        backend.model.eval()

        with torch.inference_mode():
            debug_inputs = backend.make_inputs(
                debug_samples
            )
            debug_logits, debug_positions = (
                backend.forward_ab_logits(
                    debug_inputs
                )
            )
            debug_probabilities = (
                backend.raw_probabilities(
                    debug_logits
                )
            )

        logger.info(
            "DEBUG FIRST BATCH | labels=%s "
            "valid_lengths=%s positions=%s "
            "logits=%s p_like=%s",
            [
                sample.label
                for sample in debug_samples
            ],
            (
                debug_inputs["attention_mask"]
                .sum(dim=1)
                .detach()
                .cpu()
                .tolist()
            ),
            (
                debug_positions.detach()
                .cpu()
                .tolist()
            ),
            (
                debug_logits.detach()
                .cpu()
                .tolist()
            ),
            (
                debug_probabilities.detach()
                .cpu()
                .tolist()
            ),
        )

        del (
            debug_inputs,
            debug_logits,
            debug_positions,
            debug_probabilities,
        )

    global_step = 0
    stopped_early = False

    try:
        for epoch in range(
            1,
            maximum_epochs + 1,
        ):
            epoch_started = (
                time.perf_counter()
            )

            logger.info(
                "========== EPOCH %d/%d ==========",
                epoch,
                maximum_epochs,
            )

            epoch_samples = list(
                train_samples
            )
            random.Random(
                seed + epoch
            ).shuffle(epoch_samples)

            backend.model.train()
            optimizer.zero_grad(
                set_to_none=True
            )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            epoch_optimizer_steps = 0
            last_gradient_norm = 0.0
            last_learning_rate_used = (
                get_optimizer_learning_rate(
                    optimizer
                )
            )

            for (
                batch_index,
                batch_samples,
            ) in enumerate(
                batched(
                    epoch_samples,
                    batch_size,
                ),
                start=1,
            ):
                accumulation_group_start = (
                    (
                        batch_index - 1
                    )
                    // gradient_accumulation_steps
                ) * gradient_accumulation_steps + 1

                accumulation_group_end = min(
                    accumulation_group_start
                    + gradient_accumulation_steps
                    - 1,
                    batches_per_epoch,
                )

                accumulation_group_size = (
                    accumulation_group_end
                    - accumulation_group_start
                    + 1
                )

                inputs = backend.make_inputs(
                    batch_samples
                )
                targets = backend.make_targets(
                    batch_samples
                )
                ab_logits, _ = (
                    backend.forward_ab_logits(
                        inputs
                    )
                )

                classification_loss = (
                    backend.weighted_cross_entropy(
                        ab_logits,
                        targets,
                        negative_weight,
                    )
                )

                loss_components = (
                    build_margin_regularized_loss(
                        classification_loss=(
                            classification_loss
                        ),
                        ab_logits=ab_logits,
                        margin_l2_weight=(
                            margin_l2_weight
                        ),
                    )
                )
                loss = (
                    loss_components.total_loss
                )

                (
                    loss
                    / accumulation_group_size
                ).backward()

                actual_batch_size = len(
                    batch_samples
                )

                epoch_loss_sum += (
                    float(loss.item())
                    * actual_batch_size
                )
                epoch_sample_count += (
                    actual_batch_size
                )

                if (
                    batch_index
                    == accumulation_group_end
                ):
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

                    learning_rate_used = (
                        get_optimizer_learning_rate(
                            optimizer
                        )
                    )
                    last_learning_rate_used = (
                        learning_rate_used
                    )

                    optimizer.step()
                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    global_step += 1
                    epoch_optimizer_steps += 1

                    next_learning_rate = (
                        warmup_controller
                        .advance_after_optimizer_step(
                            optimizer=optimizer,
                            completed_optimizer_steps=(
                                global_step
                            ),
                        )
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
                        mean_probability = float(
                            backend.raw_probabilities(
                                ab_logits.detach()
                            )
                            .mean()
                            .item()
                        )

                        logger.info(
                            "epoch=%d step=%d "
                            "batch=%d/%d "
                            "samples=%d/%d "
                            "loss=%.6f avg_loss=%.6f "
                            "grad_norm=%.6f "
                            "lr_used=%.10g "
                            "next_lr=%.10g "
                            "warmup_complete=%s "
                            "batch_mean_p_like_raw=%.6f",
                            epoch,
                            global_step,
                            batch_index,
                            batches_per_epoch,
                            epoch_sample_count,
                            len(epoch_samples),
                            float(loss.item()),
                            (
                                epoch_loss_sum
                                / epoch_sample_count
                            ),
                            last_gradient_norm,
                            learning_rate_used,
                            next_learning_rate,
                            warmup_controller.is_complete(
                                global_step
                            ),
                            mean_probability,
                        )

                del (
                    inputs,
                    targets,
                    ab_logits,
                    classification_loss,
                    loss_components,
                    loss,
                )

            if torch.cuda.is_available():
                peak_memory_gib = float(
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                )
                torch.cuda.empty_cache()
            else:
                peak_memory_gib = None

            logger.info(
                "开始验证 Epoch %d",
                epoch,
            )

            validation_metrics, validation_predictions = (
                evaluate_samples(
                    backend=backend,
                    samples=val_samples,
                    batch_size=eval_batch_size,
                    negative_weight=negative_weight,
                    threshold=float(
                        evaluation_config[
                            "threshold"
                        ]
                    ),
                    ece_bins=int(
                        evaluation_config[
                            "ece_bins"
                        ]
                    ),
                )
            )

            monitor_raw_value = (
                validation_metrics.get(
                    monitor_name
                )
            )
            monitor_available = (
                is_valid_monitor_value(
                    monitor_raw_value
                )
            )

            learning_rate_before_plateau = (
                get_optimizer_learning_rate(
                    optimizer
                )
            )

            plateau_result: Dict[str, Any] = {
                "skipped": True,
                "reason": None,
                "improved": None,
                "best_value": (
                    plateau_controller.best_value
                ),
                "bad_epochs": (
                    plateau_controller.bad_epochs
                ),
                "reduced": False,
                "reduction_count": (
                    plateau_controller.reduction_count
                ),
                "old_learning_rate": (
                    learning_rate_before_plateau
                ),
                "new_learning_rate": (
                    learning_rate_before_plateau
                ),
            }

            early_result: Dict[str, Any] = {
                "skipped": True,
                "reason": None,
                "improved": None,
                "best_value": (
                    early_stopping_controller.best_value
                ),
                "best_epoch": (
                    early_stopping_controller.best_epoch
                ),
                "bad_epochs": (
                    early_stopping_controller.bad_epochs
                ),
                "should_stop": False,
            }

            if not monitor_available:
                plateau_result["reason"] = (
                    f"{monitor_name} 不可用"
                )
                early_result["reason"] = (
                    f"{monitor_name} 不可用"
                )

                logger.warning(
                    "Epoch %d 的 %s 不可用，"
                    "跳过学习率调整和早停判断",
                    epoch,
                    monitor_name,
                )
            else:
                monitor_value = float(
                    monitor_raw_value
                )

                early_result = (
                    early_stopping_controller.step(
                        epoch=epoch,
                        value=monitor_value,
                    )
                )
                early_result["skipped"] = False
                early_result["reason"] = None

                if warmup_controller.is_complete(
                    global_step
                ):
                    plateau_result = (
                        plateau_controller.step(
                            optimizer=optimizer,
                            value=monitor_value,
                        )
                    )
                    plateau_result[
                        "skipped"
                    ] = False
                    plateau_result["reason"] = None
                else:
                    plateau_result["reason"] = (
                        "Warmup 尚未完成"
                    )

            learning_rate_after_plateau = (
                get_optimizer_learning_rate(
                    optimizer
                )
            )

            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": (
                    epoch_loss_sum
                    / epoch_sample_count
                ),
                "train_samples": (
                    epoch_sample_count
                ),
                "optimizer_steps": (
                    epoch_optimizer_steps
                ),
                "last_gradient_norm": (
                    last_gradient_norm
                ),
                "last_learning_rate_used": (
                    last_learning_rate_used
                ),
                "learning_rate_before_plateau": (
                    learning_rate_before_plateau
                ),
                "learning_rate_after_plateau": (
                    learning_rate_after_plateau
                ),
                "warmup_complete": (
                    warmup_controller.is_complete(
                        global_step
                    )
                ),
                "peak_memory_gib": (
                    peak_memory_gib
                ),
                "elapsed_seconds": (
                    time.perf_counter()
                    - epoch_started
                ),
                "monitor_name": monitor_name,
                "monitor_value": (
                    float(monitor_raw_value)
                    if monitor_available
                    else None
                ),
                "plateau": plateau_result,
                "early_stopping": early_result,
                **validation_metrics,
            }

            epoch_dir = (
                output_dir
                / f"epoch_{epoch}"
            )

            backend.save_adapter(
                epoch_dir
            )

            write_json_atomic(
                epoch_record,
                epoch_dir
                / "epoch_metrics.json",
            )
            write_jsonl_atomic(
                validation_predictions,
                epoch_dir
                / "val_predictions.jsonl",
            )
            write_json_atomic(
                {
                    "epoch": epoch,
                    "base_model_path": str(
                        resolved_paths[
                            "model_path"
                        ]
                    ),
                    "processor_path": str(
                        processor_dir
                    ),
                    "adapter_path": str(
                        epoch_dir
                    ),
                    "max_pixels": int(
                        model_config[
                            "max_pixels"
                        ]
                    ),
                    "token_ids": {
                        "A": backend.token_a,
                        "B": backend.token_b,
                    },
                    "mapping": {
                        "A": "like",
                        "B": "dislike",
                    },
                    "negative_weight": (
                        negative_weight
                    ),
                    "prompt": prompt_config,
                },
                epoch_dir
                / "checkpoint_config.json",
            )

            is_best_epoch = bool(
                not early_result.get(
                    "skipped",
                    True,
                )
                and early_result.get(
                    "improved",
                    False,
                )
            )

            if is_best_epoch:
                best_record = {
                    "epoch": epoch,
                    "monitor": monitor_name,
                    "mode": str(
                        early_stopping_config[
                            "mode"
                        ]
                    ),
                    "value": float(
                        monitor_raw_value
                    ),
                    "adapter_path": str(
                        epoch_dir
                    ),
                    "epoch_metrics_path": str(
                        epoch_dir
                        / "epoch_metrics.json"
                    ),
                    "updated_at": (
                        datetime.now().isoformat(
                            timespec="seconds"
                        )
                    ),
                }

                write_json_atomic(
                    best_record,
                    best_checkpoint_path,
                )

                history["best_epoch"] = epoch
                history["best_value"] = float(
                    monitor_raw_value
                )

                logger.info(
                    "更新最佳 checkpoint："
                    "epoch=%d，%s=%.6f",
                    epoch,
                    monitor_name,
                    float(monitor_raw_value),
                )

            history["epochs"].append(
                epoch_record
            )
            history[
                "last_completed_epoch"
            ] = epoch
            history["global_step"] = (
                global_step
            )

            write_json_atomic(
                history,
                history_path,
            )

            logger.info(
                "EPOCH %d | train_loss=%.6f "
                "val_auc=%s "
                "val_bal_acc=%.6f "
                "val_pr_negative=%s "
                "val_brier=%.6f "
                "val_ece=%.6f "
                "margin=%s "
                "lr=%.10g "
                "lr_reduced=%s "
                "early_bad_epochs=%s",
                epoch,
                epoch_record[
                    "train_loss"
                ],
                epoch_record[
                    "roc_auc"
                ],
                epoch_record[
                    "balanced_accuracy"
                ],
                epoch_record[
                    "pr_auc_negative"
                ],
                epoch_record[
                    "brier"
                ],
                epoch_record[
                    "ece"
                ],
                epoch_record[
                    "mean_margin_positive_minus_negative"
                ],
                learning_rate_after_plateau,
                plateau_result.get(
                    "reduced",
                    False,
                ),
                early_result.get(
                    "bad_epochs"
                ),
            )

            if bool(
                early_result.get(
                    "should_stop",
                    False,
                )
            ):
                stopped_early = True

                history["status"] = (
                    "early_stopped"
                )
                history["stop_reason"] = {
                    "epoch": epoch,
                    "monitor": monitor_name,
                    "best_epoch": (
                        early_stopping_controller
                        .best_epoch
                    ),
                    "best_value": (
                        early_stopping_controller
                        .best_value
                    ),
                    "bad_epochs": (
                        early_stopping_controller
                        .bad_epochs
                    ),
                    "patience": int(
                        early_stopping_config[
                            "patience"
                        ]
                    ),
                }

                write_json_atomic(
                    history,
                    history_path,
                )

                logger.info(
                    "触发 Early Stopping："
                    "epoch=%d，最佳 epoch=%s，"
                    "最佳 %s=%s",
                    epoch,
                    early_stopping_controller
                    .best_epoch,
                    monitor_name,
                    early_stopping_controller
                    .best_value,
                )
                break

        if not stopped_early:
            history["status"] = "completed"

        history["finished_at"] = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )

        write_json_atomic(
            history,
            history_path,
        )

        logger.info(
            "训练结束，status=%s，输出目录：%s",
            history["status"],
            output_dir,
        )

    except Exception as exc:
        history["status"] = "failed"
        history["error"] = repr(exc)
        history["global_step"] = (
            global_step
        )
        history["finished_at"] = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )

        write_json_atomic(
            history,
            history_path,
        )

        logger.exception("训练失败")
        raise
