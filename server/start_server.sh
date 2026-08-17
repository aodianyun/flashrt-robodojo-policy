#!/usr/bin/env bash
# GOAI FlashRT — WS policy server 启动脚本
#
# 全部业务参数用 CLI 参数控制；环境变量仅保留系统层
# （XLA 内存、tokenizer 路径等由 server 脚本内部自动设置）。
#
# 用法:
#   bash server/start_server.sh [--checkpoint PATH] [--model NAME] [--ckpt-dir DIR] \
#       [--framework jax|torch] [--quantization fp8|fp4|fp4-awq|bf16] \
#       [--host 0.0.0.0] [--port N] [--num-views 3] [--action-dim 14] \
#       [--download-missing]
#
# 权重指定两种方式:
#   1) --checkpoint PATH  显式路径
#   2) --model NAME       从 <ckpt-dir>/<name>/ 自动解析 (默认 pi05-arx-x5)
#
# 示例:
#   bash server/start_server.sh --quantization fp8          # FP8 (默认)
#   bash server/start_server.sh --quantization fp4-awq      # NVFP4 + AWQ
#   bash server/start_server.sh --framework torch --quantization bf16 --model pi05-arx-x5
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT=""
MODEL="pi05-arx-x5"
CKPT_DIR="/data/ckpts"
DOWNLOAD_MISSING=""
PORT="3001"
HOST="0.0.0.0"
FRAMEWORK="jax"
QUANT="fp8"
NUM_VIEWS="3"
ACTION_DIM="14"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)       CHECKPOINT="$2"; shift 2 ;;
    --model)            MODEL="$2"; shift 2 ;;
    --ckpt-dir)         CKPT_DIR="$2"; shift 2 ;;
    --download-missing) DOWNLOAD_MISSING="--download-missing"; shift ;;
    --port)             PORT="$2"; shift 2 ;;
    --host)             HOST="$2"; shift 2 ;;
    --framework)        FRAMEWORK="$2"; shift 2 ;;
    --quantization)     QUANT="$2"; shift 2 ;;
    --num-views)        NUM_VIEWS="$2"; shift 2 ;;
    --action-dim)       ACTION_DIM="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--checkpoint PATH] [--model NAME] [--ckpt-dir DIR]"
      echo "          [--framework jax|torch] [--quantization fp8|fp4|fp4-awq|bf16]"
      echo "          [--host IP] [--port N] [--num-views N] [--action-dim N] [--download-missing]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# 系统层环境（非业务参数）
export PYTHONPATH="${PROJECT_ROOT}/vendor/FlashRT:${PROJECT_ROOT}/vendor/xp_lib:${PROJECT_ROOT}"
export GOAI_PROJECT_ROOT="${PROJECT_ROOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export FLASHRT_PI05_STATE_PROMPT_MODE=fixed
# tokenizer model 随项目自带，避免首次联网下载
export FLASH_RT_PALIGEMMA_TOKENIZER="${PROJECT_ROOT}/vendor/FlashRT/assets/paligemma_tokenizer.model"
# Python 解释器: 默认系统 python3；部署时可用 venv python (deploy.sh 传入)
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG=/tmp/ws_server_${PORT}.log
PIDFILE=/tmp/ws_server_${PORT}.pid

# _alive PID: 0 if not running; zombie (Z) counts as gone (already exited).
_alive() {
  [ -d "/proc/$1" ] || return 1
  local st
  st=$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)
  [ -n "$st" ] && [ "$st" != "Z" ] || return 1
  return 0
}

# Gracefully stop any existing instance (SIGTERM, wait for clean exit).
if [ -f "$PIDFILE" ]; then
  OLD=$(cat "$PIDFILE")
  if _alive "$OLD"; then
    echo "==[GOAI] stopping existing server pid=$OLD =="
    kill -TERM "$OLD" 2>/dev/null || true
    # wait up to 15s for clean exit
    for _ in $(seq 1 30); do
      _alive "$OLD" || break
      sleep 0.5
    done
    if _alive "$OLD"; then
      echo "[WARN] server did not exit cleanly; forcing kill" >&2
      kill -KILL "$OLD" 2>/dev/null || true
    fi
  fi
  rm -f "$PIDFILE"
fi

echo "==[GOAI FlashRT server]=="
if [ -n "${CHECKPOINT}" ]; then
  echo "  checkpoint: ${CHECKPOINT}"
else
  echo "  model:      ${MODEL}   ckpt-dir: ${CKPT_DIR}"
  [ -n "${DOWNLOAD_MISSING}" ] && echo "  download:   auto (missing weights)"
fi
echo "  framework:  ${FRAMEWORK}   quantization: ${QUANT}"
echo "  host:port:  ${HOST}:${PORT}   num_views: ${NUM_VIEWS}   action_dim: ${ACTION_DIM}"
echo "  FlashRT:    ${PROJECT_ROOT}/vendor/FlashRT"
echo "  fixed mode: ${FLASHRT_PI05_STATE_PROMPT_MODE}"

# Start fully detached
# 权重解析: 显式 checkpoint 或 <ckpt-dir>/<model>/
ARGS=(--port "${PORT}" --host "${HOST}" --framework "${FRAMEWORK}"
      --quantization "${QUANT}" --num-views "${NUM_VIEWS}" --action-dim "${ACTION_DIM}")
if [ -n "${CHECKPOINT}" ]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
else
  ARGS+=(--model "${MODEL}" --ckpt-dir "${CKPT_DIR}")
  [ -n "${DOWNLOAD_MISSING}" ] && ARGS+=(--download-missing)
fi

nohup "${PYTHON_BIN}" -u -X faulthandler "${SCRIPT_DIR}/run_server.py" \
    "${ARGS[@]}" \
    > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "started pid=$PID log=$LOG"
