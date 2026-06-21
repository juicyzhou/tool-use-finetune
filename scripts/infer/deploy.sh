#!/usr/bin/env bash
# 部署 LoRA 为 OpenAI 兼容 API
#
# 用法：
#   bash scripts/infer/deploy.sh
#   bash scripts/infer/deploy.sh --background

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../config.sh
source "${SCRIPT_DIR}/../config.sh"

BACKGROUND=0
for arg in "$@"; do
    case "${arg}" in
        --background|-b) BACKGROUND=1 ;;
    esac
done

LOG_DIR="${REPO_ROOT}/outputs/deploy"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/deploy-$(date +%Y%m%d-%H%M%S).log"
PID_FILE="${LOG_DIR}/deploy.pid"

echo "[INFO] 基座模型: ${MODEL_PATH}"
echo "[INFO] LoRA 权重: ${ADAPTER_PATH}"
echo "[INFO] API 地址: http://${DEPLOY_HOST}:${DEPLOY_PORT}/v1"

run_deploy() {
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    swift deploy \
        --model "${MODEL_PATH}" \
        --adapters "${ADAPTER_PATH}" \
        --agent_template "${AGENT_TEMPLATE}" \
        --torch_dtype bfloat16 \
        --infer_backend transformers \
        --host "${DEPLOY_HOST}" \
        --port "${DEPLOY_PORT}" \
        --served_model_name "${SERVED_MODEL_NAME}" \
        --max_new_tokens 2048 \
        --temperature 0.1
}

if [ "${BACKGROUND}" -eq 1 ] && [ -z "${DEPLOY_WORKER:-}" ]; then
    echo "[INFO] 后台启动 API 服务..."
    DEPLOY_WORKER=1 nohup bash "$0" >> "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "[INFO] PID: $(cat "${PID_FILE}")"
    echo "[INFO] 健康检查: curl http://127.0.0.1:${DEPLOY_PORT}/v1/models"
    exit 0
fi

run_deploy
