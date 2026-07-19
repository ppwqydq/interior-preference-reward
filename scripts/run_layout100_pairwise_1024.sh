#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/qwen_pref_reward"
OUTPUT_ROOT="/root/autodl-tmp/qwen_pref_reward_outputs"
LOG_ROOT="${PROJECT_ROOT}/logs"

P1_NAME="qwen3_vl_8b_layout100_pairwise_p1_ab_1024"
P2_NAME="qwen3_vl_8b_layout100_pairwise_p2_scalar_1024"

P1_CONFIG="${PROJECT_ROOT}/configs/qwen8b_layout100_pairwise_p1_ab_1024.yaml"
P2_CONFIG="${PROJECT_ROOT}/configs/qwen8b_layout100_pairwise_p2_scalar_1024.yaml"

P1_OUTPUT="${OUTPUT_ROOT}/${P1_NAME}"
P2_OUTPUT="${OUTPUT_ROOT}/${P2_NAME}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate reward

cd "${PROJECT_ROOT}"

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

timestamp="$(date +%Y%m%d_%H%M%S)"

run_experiment() {
    local name="$1"
    local config="$2"
    local output="$3"
    local log="${LOG_ROOT}/${name}_${timestamp}.log"

    if [[ -e "${output}" ]]; then
        echo "[ERROR] 输出目录已经存在：${output}"
        echo "为避免覆盖旧结果，脚本停止。"
        echo "确认确实要重跑后，请手工改名或删除该目录。"
        exit 2
    fi

    echo
    echo "============================================================"
    echo "开始实验：${name}"
    echo "配置：${config}"
    echo "输出：${output}"
    echo "日志：${log}"
    echo "开始时间：$(date -Is)"
    echo "============================================================"

    python -u scripts/train_pairwise.py \
        --config "${config}" \
        --epochs 40 \
        --output_dir "${output}" \
        --experiment_name "${name}" \
        2>&1 | tee "${log}"

    status="${PIPESTATUS[0]}"

    if [[ "${status}" -ne 0 ]]; then
        echo "[ERROR] ${name} 失败，退出码=${status}"
        exit "${status}"
    fi

    echo
    echo "${name} 训练结束：$(date -Is)"

    if [[ -f "${output}/best_checkpoint.json" ]]; then
        echo "最佳 Checkpoint："
        cat "${output}/best_checkpoint.json"
    else
        echo "[WARN] 没找到 best_checkpoint.json"
    fi
}

run_experiment \
    "${P1_NAME}" \
    "${P1_CONFIG}" \
    "${P1_OUTPUT}"

# 确保 P1 进程结束后显存被操作系统释放
sleep 10
nvidia-smi

run_experiment \
    "${P2_NAME}" \
    "${P2_CONFIG}" \
    "${P2_OUTPUT}"

echo
echo "============================================================"
echo "P1/P2 1024 全部完成：$(date -Is)"
echo "P1：${P1_OUTPUT}"
echo "P2：${P2_OUTPUT}"
echo "============================================================"
