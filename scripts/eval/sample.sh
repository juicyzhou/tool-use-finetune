#!/usr/bin/env bash
# 在样本数据上批量推理（快速查看生成效果）
#
# 用法：
#   bash scripts/eval/sample.sh
#   VAL_SAMPLES=50 bash scripts/eval/sample.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../config.sh
source "${SCRIPT_DIR}/../config.sh"

SAMPLE_FILE="${SAMPLE_FILE:-${SAMPLE_DATASET}}"
VAL_SAMPLES="${VAL_SAMPLES:-20}"
RESULT_DIR="${EVAL_OUT_DIR}"
mkdir -p "${RESULT_DIR}"
RESULT_PATH="${RESULT_DIR}/infer-$(date +%Y%m%d-%H%M%S).jsonl"

EVAL_FILE="${RESULT_DIR}/eval-${VAL_SAMPLES}.jsonl"
head -n "${VAL_SAMPLES}" "${SAMPLE_FILE}" > "${EVAL_FILE}"

echo "[INFO] 基座模型: ${MODEL_PATH}"
echo "[INFO] LoRA 权重: ${ADAPTER_PATH}"
echo "[INFO] 评测样本: ${EVAL_FILE} (${VAL_SAMPLES} 条)"
echo "[INFO] 结果输出: ${RESULT_PATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
swift infer \
    --model "${MODEL_PATH}" \
    --adapters "${ADAPTER_PATH}" \
    --agent_template "${AGENT_TEMPLATE}" \
    --torch_dtype bfloat16 \
    --infer_backend transformers \
    --val_dataset "${EVAL_FILE}" \
    --result_path "${RESULT_PATH}" \
    --max_new_tokens 1024 \
    --temperature 0.1

echo "[INFO] 完成: ${RESULT_PATH}"
