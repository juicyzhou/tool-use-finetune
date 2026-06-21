#!/usr/bin/env bash
# 统一入口：bash scripts/run.sh <command> [args...]
#
#   train          LoRA 微调（推荐）
#   train-full     全量数据 / 后台 / 本地 SwanLab
#   convert        Glaive → ms-swift 格式转换
#   stats          Token 长度分布统计
#   eval           L1 主评测：工具名 + 参数 F1
#   eval-sample    L2 批量推理看样例
#   eval-cases     L3 手工 case 交互验证
#   chat           交互式 tool 循环 Demo
#   deploy         OpenAI 兼容 API 部署

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

usage() {
    cat <<'EOF'
命令：
  train          LoRA 微调（推荐）
  train-full     全量数据 / 后台 / 本地 SwanLab
  convert        Glaive → ms-swift 格式转换
  stats          Token 长度分布统计
  eval           L1 主评测：工具名 + 参数 F1
  eval-sample    L2 批量推理看样例
  eval-cases     L3 手工 case 交互验证
  chat           交互式 tool 循环 Demo
  deploy         OpenAI 兼容 API 部署
EOF
    echo ""
    echo "示例："
    echo "  bash scripts/run.sh train --gpus 0,1"
    echo "  bash scripts/run.sh eval --max-samples 500"
    echo "  bash scripts/run.sh chat"
    exit "${1:-0}"
}

CMD="${1:-}"
shift || true

case "${CMD}" in
    train)
        exec bash "${SCRIPT_DIR}/train.sh" "$@"
        ;;
    train-full)
        exec bash "${SCRIPT_DIR}/train_full.sh" "$@"
        ;;
    convert)
        exec python3 "${REPO_ROOT}/data_process/glaive_to_ms_swift.py" "$@"
        ;;
    stats)
        exec python3 "${SCRIPT_DIR}/stats_token_length.py" "$@"
        ;;
    eval)
        exec python3 "${SCRIPT_DIR}/eval/tool_selection.py" "$@"
        ;;
    eval-sample)
        exec bash "${SCRIPT_DIR}/eval/sample.sh" "$@"
        ;;
    eval-cases)
        exec python3 "${SCRIPT_DIR}/eval/run_cases.py" "$@"
        ;;
    chat)
        exec python3 "${SCRIPT_DIR}/infer/chat_agent.py" "$@"
        ;;
    deploy)
        exec bash "${SCRIPT_DIR}/infer/deploy.sh" "$@"
        ;;
    help|-h|--help|"")
        usage 0
        ;;
    *)
        echo "[ERROR] 未知命令: ${CMD}" >&2
        usage 1
        ;;
esac
