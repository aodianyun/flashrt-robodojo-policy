#!/usr/bin/env bash
# XPolicyLab 标准策略服务器启动脚本 (GOAI 2026 / flashrt-robodojo-policy)
#
# 与 RoboDojo 评测端配套:
#   bash scripts/robodojo.sh server --policy-dir XPolicyLab/policy/goai_flashrt ...
#   bash scripts/robodojo.sh client --policy-host <IP> --policy-port <PORT> ...
#
# 用法 (对齐 XPolicyLab setup_eval_policy_server.sh 语义):
#   bash setup_eval_policy_server.sh \
#       [--checkpoint PATH] [--port N] [--framework jax|torch] \
#       [--quantization fp8|fp4|fp4-awq|bf16] [--hardware auto|thor|rtx_sm120|rtx_sm89] \
#       [--num-views N] [--action-dim N] [--host IP]
#
# 默认: checkpoint=/models/model, port=3101, framework=jax, quant=fp8,
#       hardware=auto, num-views=3, action-dim=14, host=0.0.0.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# goai_flashrt -> policy -> XPolicyLab -> xp_lib -> vendor -> repo root
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
# 解释器: 优先 PYTHON_BIN (可注入), 否则 PATH 中的 python3
PYTHON_BIN="${PYTHON_BIN:-python3}"

CHECKPOINT="${CHECKPOINT:-/models/model}"
PORT="${PORT:-3101}"
FRAMEWORK="${FRAMEWORK:-jax}"
QUANT="${QUANTIZATION:-fp8}"
HARDWARE="${HARDWARE:-auto}"
NUM_VIEWS="${NUM_VIEWS:-3}"
ACTION_DIM="${ACTION_DIM:-14}"
HOST="${HOST:-0.0.0.0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    --framework)    FRAMEWORK="$2"; shift 2 ;;
    --quantization) QUANT="$2"; shift 2 ;;
    --hardware)     HARDWARE="$2"; shift 2 ;;
    --num-views)    NUM_VIEWS="$2"; shift 2 ;;
    --action-dim)   ACTION_DIM="$2"; shift 2 ;;
    --host)         HOST="$2"; shift 2 ;;
    *) # XPolicyLab 位置参数 (bench task ckpt env action seed ...) 忽略
       shift ;;
  esac
done

echo "== GOAI FlashRT 策略服务器 =="
echo "  checkpoint: ${CHECKPOINT}"
echo "  host:port:  ${HOST}:${PORT}  framework=${FRAMEWORK} quant=${QUANT} hardware=${HARDWARE}"
echo "  num_views:  ${NUM_VIEWS}  action_dim: ${ACTION_DIM}"

export PYTHONPATH="${PROJECT_ROOT}/vendor/FlashRT:${PROJECT_ROOT}/vendor/xp_lib:${PROJECT_ROOT}"
export GOAI_PROJECT_ROOT="${PROJECT_ROOT}"
export FLASHRT_PI05_STATE_PROMPT_MODE="${FLASHRT_PI05_STATE_PROMPT_MODE:-fixed}"
export FLASH_RT_PALIGEMMA_TOKENIZER="${FLASH_RT_PALIGEMMA_TOKENIZER:-${PROJECT_ROOT}/vendor/FlashRT/assets/paligemma_tokenizer.model}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

exec "${PYTHON_BIN}" -u "${PROJECT_ROOT}/server/run_server.py" \
  --checkpoint "${CHECKPOINT}" \
  --port "${PORT}" --host "${HOST}" \
  --framework "${FRAMEWORK}" --quantization "${QUANT}" --hardware "${HARDWARE}" \
  --num-views "${NUM_VIEWS}" --action-dim "${ACTION_DIM}"
