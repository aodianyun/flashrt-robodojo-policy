#!/usr/bin/env bash
# GOAI FlashRT — 一键部署脚本（新环境）
#
# 用法:
#   bash deploy.sh [--framework jax|torch] [--venv PATH] \
#                  [--model NAME] [--ckpt-dir DIR] [--checkpoint PATH] \
#                  [--quantization fp8|fp4|fp4-awq|bf16] [--port N] [--download] \
#                  [--hardware auto|thor|rtx_sm120|rtx_sm89]
#
# 默认: framework=jax, venv=<project>/vendor/venv-jax/.venv,
#       model=pi05-arx-x5, ckpt-dir=/data/ckpts, quantization=fp8, port=3001,
#       hardware=auto (自动检测 GPU)
#
# 权重: 从 <ckpt-dir>/<model>/ 自动解析；加 --download 缺失时自动下载
#
# 流程:
#   1. 检查系统依赖 (Python/CUDA/CMake)
#   2. 创建/复用独立 venv (jax 或 torch)
#   3. 编译 FlashRT 内核 .so (若目标架构不匹配)
#   4. 下载模型权重 (若不存在)
#   5. 用 venv 的 python 启动 WS server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

FRAMEWORK="jax"
VENV_DIR=""
CHECKPOINT=""
MODEL="pi05-arx-x5"
CKPT_DIR="/data/ckpts"
DOWNLOAD="0"
QUANT="fp8"
PORT="3001"
HARDWARE="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --framework)    FRAMEWORK="$2"; shift 2 ;;
    --venv)         VENV_DIR="$2"; shift 2 ;;
    --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --ckpt-dir)     CKPT_DIR="$2"; shift 2 ;;
    --download)     DOWNLOAD="1"; shift ;;
    --quantization) QUANT="$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    --hardware)     HARDWARE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--framework jax|torch] [--venv PATH]"
      echo "          [--model NAME] [--ckpt-dir DIR] [--checkpoint PATH]"
      echo "          [--quantization fp8|fp4|fp4-awq|bf16] [--port N] [--download]"
      echo "          [--hardware auto|thor|rtx_sm120|rtx_sm89]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "==== GOAI FlashRT 部署 ===="
echo "  framework:    ${FRAMEWORK}"
echo "  quantization: ${QUANT}"
echo "  hardware:     ${HARDWARE}"
if [ -n "${CHECKPOINT}" ]; then
  echo "  checkpoint:   ${CHECKPOINT}"
else
  echo "  model:        ${MODEL}   ckpt-dir: ${CKPT_DIR}   download: ${DOWNLOAD}"
fi
echo "  port:         ${PORT}"

# ── 1. 系统依赖检查 ──
echo ""
echo "== [1/5] 系统依赖检查 =="
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python: ${PY_VER}"
command -v cmake >/dev/null && echo "  cmake: $(cmake --version | head -1)" || echo "  cmake: 缺失 (编译FlashRT时需要)"
command -v nvcc >/dev/null && echo "  CUDA: $(nvcc --version | tail -1)" || echo "  CUDA: nvcc 未找到 (需要 CUDA toolkit)"

# ── 2. venv ──
echo ""
echo "== [2/5] venv (${FRAMEWORK}) =="
if [ -z "${VENV_DIR}" ]; then
  VENV_DIR="${PROJECT_ROOT}/vendor/venv-${FRAMEWORK}/.venv"
fi
if [ -x "${VENV_DIR}/bin/python" ]; then
  echo "  复用 venv: ${VENV_DIR}"
else
  echo "  创建 venv: ${VENV_DIR}"
  if [ "${FRAMEWORK}" == "torch" ]; then
    bash "${PROJECT_ROOT}/vendor/venv-torch/install_torch_venv.sh" "${VENV_DIR}"
  else
    bash "${PROJECT_ROOT}/vendor/venv-jax/install_jax_venv.sh" "${VENV_DIR}"
  fi
fi
PYTHON_BIN="${VENV_DIR}/bin/python"

# ── 3. FlashRT 内核 .so ──
echo ""
echo "== [3/5] FlashRT 内核检查 =="
SO_FILE=$(ls "${PROJECT_ROOT}"/vendor/FlashRT/flash_rt/flash_rt_kernels*.so 2>/dev/null | head -1)
ARCH_TAG="cpython-$(python3 -c 'import sys; print(sys.version_info.minor)')-$(uname -m)-linux-gnu"
if [ -n "${SO_FILE}" ] && echo "${SO_FILE}" | grep -q "${ARCH_TAG}"; then
  echo "  已存在匹配 .so (${SO_FILE})，跳过编译"
else
  echo "  .so 架构不匹配 (目标: ${ARCH_TAG})，需重新编译。"
  echo "  参考 vendor/FlashRT/README.md 准备源码 + CUTLASS 后:"
  echo "    cd vendor/FlashRT && pip install -e \".[jax]\" && cd build && cmake .. && make -j4"
fi

# ── 4. 模型权重 ──
echo ""
echo "== [4/5] 模型权重 =="
RESOLVE_ARGS=()
if [ -n "${CHECKPOINT}" ]; then
  echo "  使用显式路径: ${CHECKPOINT}"
else
  if [ "${DOWNLOAD}" == "1" ]; then
    RESOLVE_ARGS+=(--download)
  fi
  RESOLVE_ARGS+=(--format "$( [ "${FRAMEWORK}" == "torch" ] && echo safetensors || echo orbax )")
  CKPT_RESOLVED=$(bash "${PROJECT_ROOT}/scripts/resolve_checkpoint.sh" \
    "${MODEL}" "${CKPT_DIR}" "${RESOLVE_ARGS[@]}")
  echo "  权重解析: ${CKPT_RESOLVED}"
  CHECKPOINT="${CKPT_RESOLVED}"
fi

# ── 5. 启动 server（用 venv python）──
echo ""
echo "== [5/5] 启动 WS server =="
"${PYTHON_BIN}" - <<PY
import pathlib, sys
# 预检: venv 内能 import 项目模块
root = pathlib.Path("${PROJECT_ROOT}")
for p in (str(root/"vendor"/"FlashRT"), str(root/"vendor"/"xp_lib"), str(root)):
    sys.path.insert(0, p)
import flash_rt
print("  flash_rt:", flash_rt.__file__)
PY

echo "  启动 server (venv=${VENV_DIR})..."
env \
  GOAI_PROJECT_ROOT="${PROJECT_ROOT}" \
  PYTHONPATH="${PROJECT_ROOT}/vendor/FlashRT:${PROJECT_ROOT}/vendor/xp_lib:${PROJECT_ROOT}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  FLASHRT_PI05_STATE_PROMPT_MODE=fixed \
  FLASH_RT_PALIGEMMA_TOKENIZER="${PROJECT_ROOT}/vendor/FlashRT/assets/paligemma_tokenizer.model" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/server/start_server.sh" \
    --checkpoint "${CHECKPOINT}" \
    --port "${PORT}" \
    --framework "${FRAMEWORK}" \
    --quantization "${QUANT}" \
    --hardware "${HARDWARE}"

echo ""
echo "==== 部署完成 ===="
echo "  server: ws://<IP>:<公网映射端口> (port ${PORT})"
echo "  日志:   /tmp/ws_server_${PORT}.log"
