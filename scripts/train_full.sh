#!/usr/bin/env bash
# 全量数据训练（高级）：自动转换数据、后台运行、可选本地 SwanLab
#
# 用法：
#   bash scripts/train_full.sh
#   bash scripts/train_full.sh --background
#   ENABLE_SWANLAB=1 bash scripts/train_full.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

BACKGROUND=0
for arg in "$@"; do
    case "${arg}" in
        --background|-b) BACKGROUND=1 ;;
    esac
done
[ "${BACKGROUND:-0}" = "1" ] && BACKGROUND=1

FULL_DATASET="${REPO_ROOT}/data/glaive-function-calling-v2-ms-swift/glaive-ms-swift.jsonl"
DATASET="${REPO_ROOT}/data/glaive-function-calling-v2-ms-swift/glaive-ms-swift-10000.jsonl"
TRAIN_SAMPLES=10000
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUT_DIR}}"

REPORT_TO=("tensorboard")
if [ "${ENABLE_SWANLAB:-0}" = "1" ]; then
    REPORT_TO=("tensorboard" "swanlab")
    export SWANLAB_MODE=local
    SWANLAB_PROJECT="${SWANLAB_PROJECT:-tool-use}"
    SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME:-qwen3-4b-glaive}"
fi

: "${NPROC_PER_NODE:=$(nvidia-smi -L 2>/dev/null | wc -l)}"
export NPROC_PER_NODE
mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${FULL_DATASET}" ]; then
    echo "[INFO] 转换全量数据集..."
    python3 "${REPO_ROOT}/data_process/glaive_to_ms_swift.py" \
        --input  "${REPO_ROOT}/data/glaive-function-calling-v2/glaive-function-calling-v2.json" \
        --output "${FULL_DATASET}"
fi

if [ ! -f "${DATASET}" ] || [ "${FULL_DATASET}" -nt "${DATASET}" ]; then
    echo "[INFO] 截取前 ${TRAIN_SAMPLES} 条 -> ${DATASET}"
    head -n "${TRAIN_SAMPLES}" "${FULL_DATASET}" > "${DATASET}"
fi

PER_DEVICE_BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
DATASET_NUM_PROC=8
LOG_FILE="${OUTPUT_DIR}/train-$(date +%Y%m%d-%H%M%S).log"
PID_FILE="${OUTPUT_DIR}/train.pid"

if [ "${BACKGROUND}" -eq 1 ] && [ -z "${TRAIN_WORKER:-}" ]; then
    echo "[INFO] 后台启动训练..."
    TRAIN_WORKER=1 nohup bash "$0" >> "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "[INFO] PID: $(cat "${PID_FILE}")  日志: ${LOG_FILE}"
    exit 0
fi

SWANLAB_ARGS=()
if [ "${ENABLE_SWANLAB:-0}" = "1" ]; then
    SWANLAB_ARGS=(
        --swanlab_project  "${SWANLAB_PROJECT}"
        --swanlab_exp_name "${SWANLAB_EXP_NAME}"
        --swanlab_mode     local
    )
fi

swift sft \
    --model "${MODEL_PATH}" \
    --dataset "${FULL_DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --agent_template hermes \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_length 4096 \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 100 \
    --split_dataset_ratio 0.01 \
    --dataloader_num_workers 8 \
    --report_to "${REPORT_TO[@]}" \
    "${SWANLAB_ARGS[@]}" \
    --deepspeed zero3
