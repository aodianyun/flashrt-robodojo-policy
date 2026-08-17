#!/usr/bin/env bash
# GOAI FlashRT — 独立 JAX venv 创建脚本
#
# 用法: bash vendor/venv-jax/install_jax_venv.sh [venv_dir]
# 默认 venv: <project>/vendor/venv-jax/.venv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${1:-${SCRIPT_DIR}/.venv}"

echo "==== JAX venv 安装 ===="
echo "  venv: ${VENV_DIR}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

echo "== 安装 JAX 栈 (版本锁定, 对齐运行时) =="
# 平台说明: aarch64 (Thor SM110) + CUDA 13。其他平台需调整 jax-cuda13-plugin。
python -m pip install \
    jax==0.11.0 \
    jax-cuda13-pjrt==0.11.0 \
    jax-cuda13-plugin==0.11.0 \
    ml-dtypes==0.5.4 \
    orbax-checkpoint==0.12.2 \
    flax==0.12.8 \
    numpy==2.5.1

# WS server 依赖
python -m pip install websockets msgpack msgpack-numpy pyyaml sentencepiece pillow

echo "== 验证 =="
python - <<'PY'
import jax
print("jax:", jax.__version__)
try:
    print("devices:", jax.devices())
except Exception as e:
    print("devices WARN:", e)
PY

echo "==== 完成 ===="
echo "使用: source ${VENV_DIR}/bin/activate"
echo "启动 server 时该 venv 的 python 已含 jax。"
