#!/usr/bin/env bash
# Glaive Tool-Use LoRA 微调（推荐入口）
#
# 用法：
#   bash scripts/train.sh
#   bash scripts/run.sh train --gpus 0,1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

DATASET="${DATASET}"
OUTPUT_DIR="${OUTPUT_DIR}"

# batch：全局 batch = 卡数 × PER_DEVICE × GRAD_ACCUM（默认值见 config.sh）
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)     CUDA_DEVICES="$2"; shift 2 ;;
    --gpus=*)   CUDA_DEVICES="${1#*=}"; shift ;;
    *) echo "[ERROR] 未知参数: $1（支持 --gpus 0,1,2,3）" >&2; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
NUM_GPUS="$(echo "${CUDA_DEVICES}" | tr ',' '\n' | sed '/^$/d' | wc -l)"
export NPROC_PER_NODE="${NUM_GPUS}"
DATASET_NUM_PROC="${NUM_GPUS}"
[ "${DATASET_NUM_PROC}" -gt 8 ] && DATASET_NUM_PROC=8

if [ ! -f "${DATASET}" ]; then
  echo "[ERROR] 数据集不存在: ${DATASET}" >&2
  echo "请见 data/glaive-function-calling-v2-ms-swift/README.md 下载数据" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NPROC_PER_NODE=${NPROC_PER_NODE}  dataset_num_proc=${DATASET_NUM_PROC}"

swift sft \
  --model "${MODEL_PATH}" \
  --dataset "${DATASET}" \
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
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --split_dataset_ratio 0.01 \
  --report_to tensorboard \
  --deepspeed "${DEEPSPEED}"
