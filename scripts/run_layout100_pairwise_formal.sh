#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

cd /root/qwen_pref_reward

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_experiment() {
  local experiment="$1"
  local config="$2"

  local output_dir="/root/autodl-tmp/qwen_pref_reward_outputs/${experiment}"
  local log_file="logs/${experiment}_console.log"

  if [ -d "${output_dir}" ] && \
     [ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    echo "输出目录非空，为避免覆盖已停止：${output_dir}"
    exit 1
  fi

  mkdir -p "${output_dir}"

  echo
  echo "=================================================="
  echo "开始训练：${experiment}"
  echo "配置：${config}"
  echo "输出：${output_dir}"
  echo "时间：$(date '+%F %T')"
  echo "=================================================="

  python scripts/train_pairwise.py \
    --config "${config}" \
    --output_dir "${output_dir}" \
    2>&1 | tee "${log_file}"

  local best_file="${output_dir}/best_checkpoint.json"

  if [ ! -f "${best_file}" ]; then
    echo "缺少最佳 Checkpoint 记录：${best_file}"
    exit 1
  fi

  local best_checkpoint
  best_checkpoint="$(
    python - "${best_file}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
print(data["checkpoint_dir"])
PY
  )"

  echo "最佳 Checkpoint：${best_checkpoint}"

  python scripts/evaluate_pairwise_checkpoint.py \
    --checkpoint_dir "${best_checkpoint}" \
    --manifest data/splits/layout100_curated/val_pairwise.jsonl \
    --output_dir "${output_dir}/best_val_pairwise_evaluation" \
    --pair_batch_size 1 \
    --bootstrap_iterations 10000 \
    --bootstrap_seed 42 \
    2>&1 | tee \
      "${output_dir}/best_val_pairwise_evaluation.log"

  echo
  echo "完成：${experiment}"
  echo "时间：$(date '+%F %T')"
}

run_experiment \
  qwen3_vl_8b_layout100_pairwise_p1_ab_512 \
  configs/qwen8b_layout100_pairwise_p1_ab_512.yaml

run_experiment \
  qwen3_vl_8b_layout100_pairwise_p2_scalar_512 \
  configs/qwen8b_layout100_pairwise_p2_scalar_512.yaml

echo
echo "P1、P2 训练和最佳 Checkpoint 评估全部完成。"
