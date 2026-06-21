#!/usr/bin/env bash
# 全局配置（训练 / 评测 / 推理共用）
# 所有路径相对于 REPO_ROOT；可通过环境变量覆盖

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── 模型与数据 ────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/Qwen3-4B}"
DATASET="${DATASET:-${REPO_ROOT}/data/glaive-function-calling-v2-ms-swift/glaive-ms-swift-10000.jsonl}"
SAMPLE_DATASET="${SAMPLE_DATASET:-${REPO_ROOT}/data/glaive-function-calling-v2-ms-swift/glaive-ms-swift-sample-1000.jsonl}"

# ── 输出 ──────────────────────────────────────────────────────────────────────
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${REPO_ROOT}/outputs/tool-use-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUT_DIR}}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${REPO_ROOT}/outputs/eval}"

# ── 训练默认 ──────────────────────────────────────────────────────────────────
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
DEEPSPEED="${DEEPSPEED:-zero3}"
AGENT_TEMPLATE="${AGENT_TEMPLATE:-hermes}"

# ── 部署 ──────────────────────────────────────────────────────────────────────
DEPLOY_HOST="${DEPLOY_HOST:-0.0.0.0}"
DEPLOY_PORT="${DEPLOY_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-glaive-tool-use}"

resolve_latest_adapter() {
    local latest
    latest="$(ls -td "${TRAIN_OUT_DIR}"/v*/checkpoint-* 2>/dev/null | head -1 || true)"
    if [ -z "${latest}" ]; then
        echo "[ERROR] 未找到 LoRA checkpoint，请先完成训练。" >&2
        echo "  搜索路径: ${TRAIN_OUT_DIR}/v*/checkpoint-*" >&2
        return 1
    fi
    echo "${latest}"
}

resolve_latest_val_dataset() {
    local latest
    latest="$(ls -t "${TRAIN_OUT_DIR}"/v*/val_dataset.jsonl 2>/dev/null | head -1 || true)"
    if [ -z "${latest}" ]; then
        echo "[ERROR] 未找到 val_dataset.jsonl，请先完成训练。" >&2
        echo "  搜索路径: ${TRAIN_OUT_DIR}/v*/val_dataset.jsonl" >&2
        return 1
    fi
    echo "${latest}"
}

ADAPTER_PATH="${ADAPTER_PATH:-$(resolve_latest_adapter 2>/dev/null || true)}"
